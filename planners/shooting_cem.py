from tensorflow_probability.substrates import jax as tfp
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

tfd = tfp.distributions

# Adapted from: https://github.com/zchuning/latco/blob/6aab525b66efb8c99e55d6e0587a7bd31a599809/planners/shooting_cem.py


class ShootingCEM():
    def __init__(self, env, cfg, key, config_rddlsim={}):
        self.env = env
        self.alg = cfg['alg']
        self.action_keys = config_rddlsim.get('action_keys', [])
        self.const_dict = config_rddlsim.get('const_dict', {})
        self.var_dict = config_rddlsim.get('var_dict', {})
        self.reward_fn = config_rddlsim.get('reward_fn')
        self.dynamics_fn = config_rddlsim.get('transition_fn')
        self.cpfs = config_rddlsim.get('cpfs')
        
        self.nA = len(self.action_keys)
            
        self.plan_fn = self.evaluate_mppi if self.alg == "mppi" else self.evaluate_cem
        self.plan_horizon = cfg["depth"]
        self.pop_size = cfg[self.alg]["n_samples"]
        self.optimization_steps = cfg[self.alg]['optimization_steps']
        self.mppi_gamma = cfg["mppi"]["gamma"]
        self._float = jnp.float32
        self.key = key
        self.elite_size = cfg["cem"]["elite_size"]

        # This is only required for exp_samples
        if self.pop_size < self.elite_size:
            self.elite_size = self.pop_size

        self.alpha = cfg.get('alpha', 0)

        
        self.nS = cfg.get("nS", len(env.observation_space))
        self.ac_lb = np.array([env.action_space[action].low[0] for action in self.action_keys])
        self.ac_ub = np.array([env.action_space[action].high[0] for action in self.action_keys])
        self.n_noise_var = 2


        self.batch_dynamics = jax.vmap(self.dynamics_fn, in_axes=(0, 0, None), out_axes=(0, 0))
        # self.batch_rewards = jax.vmap(self.reward_fn, in_axes=(0, 0, 0 , None), out_axes=(0))
        self.batched_weighted_sample_fn = jax.vmap(lambda weight,sample: weight*sample, in_axes=(0, 0), out_axes=(0)) 

    def update_model(self, model):
        self.model = model

    def reset(self):
       self.prev_a_mean = jnp.tile((self.ac_lb + self.ac_ub)/2, [self.plan_horizon, 1])


    def eval_fitness_step(self, d, val):
        feats, actions, noise, agg_rewards = val
        current_actions = actions[d]
        current_noise = noise[d]
        feats_ = jnp.concatenate((feats, current_noise), 1)

        next_feats, rewards = self.batch_dynamics(feats_, current_actions, None)
        return next_feats, actions, noise, agg_rewards + rewards

    # Evalutes a batch of action sequences from a given obs and returns the aggregate reward for each action sequence.
    def eval_fitness(self, obs, actions, key):
        """
        obs: Vector of observed state variables
        actions: Array of action sequences. Shape: (pop_size, plan_horizon, nA)
        key: PRNG key for sampling noise
        """
        feats = jnp.tile(obs, [self.pop_size, 1])
        agg_rewards = jnp.zeros([self.pop_size], dtype=self._float)
        noise_norm = jax.random.normal(key, [self.plan_horizon, self.pop_size, 1])
        noise_uni = jax.random.uniform(key, [self.plan_horizon, self.pop_size, 1])
        noise = jnp.concatenate([noise_norm, noise_uni], axis = 2)

        # From (pop_size, plan_horizon, nA) to (plan_horizon, pop_size, nA)
        actions = actions.transpose(1, 0, 2)
        init_val = (feats, actions, noise, agg_rewards)
        # for idx in range(self.plan_horizon):
        #     action_dict = {k: v for k,v in zip(self.action_keys, actions[idx, :, :].T)}
        #     next_var_dict = self.batch_dynamics(obs_dict, action_dict, self.key)
        #     # feats = jnp.expand_dims(feats, 2)
        #     reward = self.batch_rewards(obs_dict, self.key)[0]
        #     agg_rewards += reward
        #     obs_dict =  {k: next_var_dict[f"{k}'"].reshape(-1, 1) for k in obs_dict.keys()}
        feats, _, _, agg_rewards = jax.lax.fori_loop(0, self.plan_horizon, self.eval_fitness_step, init_val)
        return agg_rewards, feats

    # samples: (batch_size, horizon, nA)
    # fitness: (batch_size,)
    @partial(jax.jit, static_argnums=(0,))
    def mppi(self, d, val):
        obs, a_mean, a_var, key = val

        # Bound variance
        lb_dist, ub_dist = a_mean - self.ac_lb, self.ac_ub - a_mean
        a_var = jnp.minimum(jnp.minimum(jnp.square(lb_dist/2), jnp.square(ub_dist/2)), a_var)
        a_std = jnp.sqrt(a_var)

        new_key, sub_key1, sub_key2 = jax.random.split(key, 3)
        # Sample action sequences and evaluate fitness
        noise = tfd.TruncatedNormal(jnp.zeros_like(a_mean), jnp.ones_like(a_var), -2, 2).sample(sample_shape=[self.pop_size], seed=sub_key1)
        samples = a_mean + noise * a_std

        fitness, _ = self.eval_fitness(obs, samples, sub_key2)

        weights = jax.nn.softmax(self.mppi_gamma * fitness)
        new_a_mean = jnp.sum(self.batched_weighted_sample_fn(weights, samples), axis=0)
        new_a_var = jnp.sum(self.batched_weighted_sample_fn(weights, jnp.square(samples - new_a_mean)), axis=0)

        return obs, new_a_mean, new_a_var, new_key

    # samples: (batch_size, horizon, nA)
    @partial(jax.jit, static_argnums=(0,))
    def cem(self, d, val):
        obs, a_mean, a_var, key = val

        # Bound variance
        lb_dist, ub_dist = a_mean - self.ac_lb, self.ac_ub - a_mean
        a_var = jnp.minimum(jnp.minimum(jnp.square(lb_dist/2), jnp.square(ub_dist/2)), a_var)
        a_std = jnp.sqrt(a_var)

        new_key, sub_key1, sub_key2 = jax.random.split(key, 3)

        # Shape: (pop_size, plan_horizon, nA)
        noise = tfd.TruncatedNormal(loc=jnp.zeros_like(a_mean), scale=jnp.ones_like(a_var), low=[-2.0], high=[2.0])

        noise = noise.sample(sample_shape=[self.pop_size], seed=sub_key1)
        # samples = jnp.tile(a_mean, [self.pop_size, 1, 1]) + noise * jnp.tile(a_std, [self.pop_size, 1, 1])
        samples = a_mean + a_std * noise
        # samples_ = tfd.MultivariateNormalDiag(means, stds).sample(sample_shape=[self.pop_size], seed=sub_key1)
        # samples = jnp.clip(samples_, self.ac_lb, self.ac_ub)
        fitness, _ = self.eval_fitness(obs, samples, sub_key2)

        # Choose elite samples and compute new means and vars
        elite_values, elite_inds = jax.lax.top_k(jnp.squeeze(fitness), self.elite_size)
        elite_samples = samples[elite_inds]
        new_a_mean = jnp.mean(elite_samples, axis=0)
        new_a_var = jnp.var(elite_samples, axis=0)

        return obs, new_a_mean, new_a_var, new_key

    def evaluate_cem(self, obs, a_mean, a_var, key):
        init_val = (obs, a_mean, a_var, key)
        # for i in range(self.optimization_steps):
        #     init_val = self.cem(i, init_val)
        # return init_val
        return jax.lax.fori_loop(0, self.optimization_steps, self.cem, init_val)

    def evaluate_mppi(self, obs, a_mean, a_var, key):
        init_val = (obs, a_mean, a_var, key)
        return jax.lax.fori_loop(0, self.optimization_steps, self.mppi, init_val)

    def choose_action(self, obs):        
        # Shape: (plan_horizon, nA)
        init_mean = self.prev_a_mean
        init_var = jnp.tile(jnp.square(self.ac_ub - self.ac_lb)/16, [self.plan_horizon, 1])

        _, mean, _, self.key = self.plan_fn(obs, init_mean, init_var, self.key)

        action, self.prev_a_mean = mean[0], jnp.concatenate((mean[1:], jnp.zeros((1, self.nA))), axis=0)

        assert (action.shape == (self.nA,))
        return action