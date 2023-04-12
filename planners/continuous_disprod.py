import jax
import jax.numpy as jnp

from planners.disprod import Disprod
from functools import partial
from utils.common_utils import random_argmax
from planners.utils import adam_with_projection
import numpy as np


class ContinuousDisprod(Disprod):
    def __init__(self, cfg, key, config_rddlsim={}):
        super(ContinuousDisprod, self).__init__(cfg, key, config_rddlsim)
    
        self.ac_lb = np.array([cfg["action_space"][a].low[0] for a in self.a_keys])
        self.ac_ub = np.array([cfg["action_space"][a].high[0] for a in self.a_keys])

        # Multiplicative factor used to transform free_action variables to the legal range.
        self.multiplicative_factor = self.ac_ub - self.ac_lb

        if cfg['disprod']['reward_fn_using_taylor']:
            self.reward_dist_fn = self.reward_comp
        else:
            self.reward_dist_fn = self.reward_mean

        self.converged_jit = jax.jit(
            lambda x, thresh: jnp.max(jnp.abs(x)) < thresh)

        self.choose_action_mean = cfg["disprod"]["choose_action_mean"]

        if cfg["disprod"]["taylor_expansion_mode"] == "complete":
            self.dynamics_dist_fn = self.dynamics_comp
        elif cfg["disprod"]["taylor_expansion_mode"] == "no_var":
            self.dynamics_dist_fn = self.dynamics_nv
        else:
            raise Exception(
                f"Unknown value for config taylor_expansion_mode. Got {cfg['taylor_expansion_mode']}")
        
        if cfg["disprod"]["choose_action_mean"]:
            self.ac_selector = lambda m,v,key: m
        else:
            self.ac_selector = lambda m,v,key: m + jnp.sqrt(v) * jax.random.normal(key, shape=(self.nA,)) 
        
        
        # Support for normal and uniform. 
        self.noise_var = 2

        self.reset()

    def reset(self):
        self.key, subkey = jax.random.split(self.key)
        self.saved_restart = jax.random.uniform(
            subkey, shape=(self.depth, self.nA))

    # osb: dict with nS - 1 keys
    def choose_action(self, obs):
        self.nS = len(obs) + self.noise_var

        # Create a vector of obs corresponding to n_restarts
        stacked_obs = jnp.tile(obs, (self.n_res, 1)).astype('float32')

        # Initialize free_ac_mean to [0,1) range and free_ac_var using a uniform distribution
        self.key, subkey1, subkey2, subkey3 = jax.random.split(self.key, 4)
        free_ac_mean = self.init_ac_mean(subkey1)
        free_ac_var = self.init_ac_var(free_ac_mean)

        opt_init_mean, self.opt_update_mean, self.get_params_mean = adam_with_projection(self.step_size)
        opt_state_mean = opt_init_mean(free_ac_mean)

        opt_init_var, self.opt_update_var, self.get_params_var = adam_with_projection(self.step_size_var)
        opt_state_var = opt_init_var(free_ac_var)

        n_grad_steps = 0
        has_converged = False

        init_val = (free_ac_mean, free_ac_var, n_grad_steps, has_converged, stacked_obs, opt_state_mean, opt_state_var, jnp.zeros((self.max_grad_steps,)))
        # Iterate until max_grad_steps reached or both means and variance has not converged
        if self.run_mode != "production":
            free_ac_mean, free_ac_var, n_grad_steps, _, _, _, _, tmp = self.update_actions(init_val)
        else:
            free_ac_mean, free_ac_var, n_grad_steps, _, _, _, _, tmp = jax.lax.while_loop(
                self.have_actions_converged, self.update_actions_optimised, init_val)

        ac_mean = self.transform_ac_mean(free_ac_mean).block_until_ready()
        ac_var = self.transform_ac_var(free_ac_var).block_until_ready()

        if self.debug:
            print(
                f"Gradients steps taken: {n_grad_steps}. Resets per step: {tmp}")

        q_value, trajectory = jax.vmap(self.q_opt, in_axes=(
            0, 0, 0), out_axes=0)(stacked_obs, ac_mean, ac_var)

        # TODO: If multiple restarts have the same q-value, should we choose the action with lowest variance?
        best_restart = random_argmax(subkey2, q_value)
        self.saved_restart = free_ac_mean[best_restart]

        print(f"Action chosen: {ac_mean[best_restart][0]}, variance: {ac_var[best_restart][0]}")
        ac = self.ac_selector(ac_mean[best_restart][0], ac_var[best_restart][0], subkey3)
        
        return jnp.clip(ac, self.ac_lb, self.ac_ub)

    @partial(jax.jit, static_argnums=(0,))
    def partials_for_exact_fn(self, operands):
        fop_wrt_state, fop_wrt_action = self.first_order_partials_for_exact_fn(operands)
        # sop_wrt_state, sop_wrt_action = self.second_order_partials_for_exact_fn(
        #     operands)
        sop_wrt_state, sop_wrt_action = self.sop_numerical(operands)
        return (fop_wrt_state, fop_wrt_action), (sop_wrt_state, sop_wrt_action)

    def first_order_partials_for_exact_fn(self, operands):
        state_means, action_means = operands
        return jax.jacfwd(self.next_state_fn, has_aux=True, argnums=(0, 1))(state_means, action_means)[0]

    def second_order_partials_for_exact_fn(self, operands):
        state_means, action_means = operands
        sop_wrt_state = self.diag_hessian_of_transition(
            state_means, action_means, 0)
        sop_wrt_action = self.diag_hessian_of_transition(
            state_means, action_means, 1)
        return sop_wrt_state, sop_wrt_action

    # def update_actions(self, val):
    #     free_action_mean, free_action_variance, n_grad_steps, has_converged, stacked_state, opt_state_mean, opt_state_var, tmp = val
    #     while n_grad_steps < self.max_grad_steps and not has_converged:
    #         # free_action_mean_old = free_action_mean.copy()
    #         # free_action_variance_old = free_action_variance.copy()

    #         # Transform action means and variance from (0,1) to permissible action ranges
    #         action_means = self.transform_ac_mean(free_action_mean)
    #         action_variance = self.transform_ac_var(free_action_variance)

    #         (reward, _), (grad_mean, grad_var) = jax.vmap(jax.value_and_grad(self.q, argnums=(
    #             1, 2), has_aux=True), in_axes=(0, 0, 0), out_axes=0)(stacked_state, action_means, action_variance)

    #         # Loss is negative of Q-value.
    #         opt_state_mean = self.opt_update_mean(
    #             n_grad_steps, -grad_mean, opt_state_mean, 0, 1)
    #         free_action_mean_ = self.get_params_mean(opt_state_mean)

    #         wiggle_room = jnp.minimum(
    #             free_action_mean_ - 0, 1 - free_action_mean_)
    #         opt_state_var = self.opt_update_var(
    #             n_grad_steps, -grad_var, opt_state_var, 0, jnp.minimum(1/12, jnp.square(wiggle_room)/12))
    #         free_action_variance_ = self.get_params_var(opt_state_var)

    #         updated_action_means = self.transform_ac_mean(free_action_mean_)
    #         updated_action_variance = self.transform_ac_var(
    #             free_action_variance_)

    #         updated_reward, _ = jax.vmap(self.q, in_axes=(0, 0, 0), out_axes=0)(
    #             stacked_state, updated_action_means, updated_action_variance)

    #         restarts_to_reset = jnp.where(updated_reward < reward, jnp.ones(
    #             self.n_res, dtype=jnp.int32), jnp.zeros(self.n_res, dtype=jnp.int32))
    #         mask = jnp.tile(restarts_to_reset, (self.depth,
    #                         self.nA, 1)).transpose(2, 0, 1)
    #         free_action_mean_final = free_action_mean * \
    #             mask + free_action_mean_ * (1-mask)
    #         free_action_variance_final = free_action_variance * \
    #             mask + free_action_variance_ * (1-mask)

    #         tmp.at[n_grad_steps].set(jnp.sum(restarts_to_reset))

    #         mean_epsilon = free_action_mean_final - free_action_mean
    #         variance_epsilon = free_action_variance_final - free_action_variance

    #         free_action_mean = free_action_mean_final
    #         free_action_variance = free_action_variance_final

    #         has_converged = jnp.max(jnp.abs(mean_epsilon)) < self.convergance_threshold and jnp.max(
    #             jnp.abs(variance_epsilon)) < self.convergance_threshold
    #         n_grad_steps += 1
    #     return free_action_mean_final, free_action_variance_final, n_grad_steps, None, None, None, tmp

    def have_actions_converged(self, val):
        _, _, n_grad_steps, has_converged, _, _, _, _ = val
        return jnp.logical_and(n_grad_steps < self.max_grad_steps, jnp.logical_not(has_converged))

    @partial(jax.jit, static_argnums=(0,))
    def transform_action_means(self, free_action_means):
        transformed_actions = self.ac_lb + self.multiplicative_factor * free_action_means
        return transformed_actions

    @partial(jax.jit, static_argnums=(0,))
    def transform_action_variance(self, free_action_variance):
        return jnp.square(self.multiplicative_factor) * free_action_variance

    def update_actions_optimised(self, val):
        free_action_mean, free_action_variance, n_grad_steps, has_converged, stacked_state, opt_state_mean, opt_state_var, tmp = val

        # # Transform action means and variance from (0,1) to permissible action ranges
        action_means = self.transform_action_means(free_action_mean)
        action_variance = self.transform_action_variance(free_action_variance)

        (reward, _), (grad_mean, grad_var) = jax.vmap(jax.value_and_grad(self.q_opt, argnums=(1, 2),
                                                                         has_aux=True), in_axes=(0, 0, 0), out_axes=0)(stacked_state, action_means, action_variance)

        # Loss is negative of Q-value.
        opt_state_mean = self.opt_update_mean(
            n_grad_steps, -grad_mean, opt_state_mean, 0, 1)
        free_action_mean_ = self.get_params_mean(opt_state_mean)

        wiggle_room = jnp.minimum(free_action_mean_ - 0, 1 - free_action_mean_)
        opt_state_var = self.opt_update_var(
            n_grad_steps, -grad_var, opt_state_var, 0, jnp.minimum(1/12, jnp.square(wiggle_room)/12))
        free_action_variance_ = self.get_params_var(opt_state_var)

        updated_action_means = self.transform_action_means(free_action_mean_)
        updated_action_variance = self.transform_action_variance(
            free_action_variance_)
        updated_reward, _ = jax.vmap(self.q_opt, in_axes=(0, 0, 0), out_axes=0)(
            stacked_state, updated_action_means, updated_action_variance)

        restarts_to_reset = jnp.where(updated_reward < reward, jnp.ones(
            self.n_res, dtype=jnp.int32), jnp.zeros(self.n_res, dtype=jnp.int32))
        mask = jnp.tile(restarts_to_reset, (self.depth,
                        self.nA, 1)).transpose(2, 0, 1)
        free_action_mean_final = free_action_mean * \
            mask + free_action_mean_ * (1-mask)
        free_action_variance_final = free_action_variance * \
            mask + free_action_variance_ * (1-mask)

        # Check for convergence of action means and variance. Changed from OR to AND
        mean_epsilon = free_action_mean_final - free_action_mean
        variance_epsilon = free_action_variance_final - free_action_variance

        has_converged = jnp.logical_and(self.converged_jit(
            mean_epsilon, self.convergance_threshold), self.converged_jit(variance_epsilon, self.convergance_threshold/10))
        return free_action_mean_final, free_action_variance_final, n_grad_steps + 1, has_converged, stacked_state, opt_state_mean, opt_state_var, tmp.at[n_grad_steps].set(jnp.sum(restarts_to_reset))

    def q_opt(self, s, a_mu, a_var):
        # augment state by adding variable for noise
        s_mu = jnp.concatenate((s, jnp.array([self.norm_noise_mu, self.uni_noise_mu])), 0)
        s_var = jnp.concatenate((s * 0, jnp.array([self.norm_noise_var, self.uni_noise_var])), 0)

        init_rew = jnp.array([0.0])
        # assert (s_mu.shape == s_var.shape == (self.nS, ))
        tau_ = jnp.zeros((2, self.depth, self.nS))
        init_params = (init_rew, s_mu, s_var, a_mu, a_var, tau_)
        agg_rew, _, _, _, _, tau = jax.lax.fori_loop( 0, self.depth, self.rollout_graph, init_params)
        return agg_rew.sum(), tau

    @partial(jax.jit, static_argnums=(0, ))
    def rollout_graph(self, d, params):
        agg_reward, s_mu, s_var, a_mu, a_var, tau = params

        # Compute next state distribution and reward
        ns_mu, ns_var, reward = self.dynamics_dist_fn(s_mu, s_var, a_mu[d, :], a_var[d, :])
    
        return agg_reward+reward, ns_mu, ns_var, a_mu, a_var, tau

    # Clamp each action between 0 and 1.
    @partial(jax.jit, static_argnums=(0,))
    def project_mean(self, free_action_mean):
        return jnp.clip(free_action_mean, 0, 1)

    # Prevent variance from becoming negative
    @partial(jax.jit, static_argnums=(0,))
    def project_variance(self, free_action_variance):
        return jnp.clip(free_action_variance, 0, 1/12)

    # Shape of FOP and SOP: (nS-1, nA), (nS-1, nS)
    @partial(jax.jit, static_argnums=(0,))
    def dynamics_comp(self, s_mu, s_var, a_mu, a_var):
        operands = s_mu, a_mu
        ns, reward = self.next_state_fn(*operands)

        (fop_w_s, fop_w_a), (sop_w_s, sop_w_a) = self.partials_for_exact_fn(operands)

        # Taylor's expansion
        ns_mu = ns + 0.5*(jnp.multiply(sop_w_a, a_var).sum(axis=1) + jnp.multiply(sop_w_s, s_var).sum(axis=1))
        ns_var = jnp.multiply(jnp.square(fop_w_a), a_var).sum(axis=1) + jnp.multiply(jnp.square(fop_w_s), s_var).sum(axis=1)

        ns_mu = jnp.concatenate([ns_mu, jnp.array([self.norm_noise_mu, self.uni_noise_mu])], axis=0)
        ns_var = jnp.concatenate([ns_var, jnp.array([self.noise_var, self.uni_noise_var])], axis=0)

        return ns_mu, ns_var, reward

    # Ignore the variance terms

    @partial(jax.jit, static_argnums=(0,))
    def dynamics_nv(self, s_mu, s_var, a_mu, a_var):

        ns = self.next_state_fn(s_mu, a_mu)
        ns_mu = jnp.concatenate([ns, jnp.array([self.norm_noise_mu, self.uni_noise_mu])], axis=0)
        ns_var = jnp.concatenate([jnp.zeros_like(ns), jnp.array([self.noise_var, self.uni_noise_var])], axis=0)

        # next_state_expectation = jnp.concatenate([next_state_expectation, jnp.array([self.noise_mean])], axis=0)
        # next_state_variance = jnp.concatenate([next_state_variance, jnp.array([self.noise_var])], axis=0)

        return ns_mu, ns_var


    @partial(jax.jit, static_argnums=(0,))
    def next_state_fn(self, state_mean, ac_mean):
        return self.dynamics_fn_wrapper(state_mean, ac_mean, None)

    def sop_numerical(self, operands):
        state_mean, ac_mean = operands
        
        # For del_s, concat extra zero rows at the bottom.
        # For del_a, concat extra zero rows at the top.
        delta = 1e-2
        # del_s = jnp.vstack([delta * jnp.eye(self.nS), jnp.zeros(self.nA, self.nS + self.nA)])
        # del_a = jnp.vstack([jnp.zeros(self.nA, self.nS + self.nA), delta * jnp.eye(self.nA)])
        
        del_sa = delta * jnp.eye(self.nS + self.nA)
        sa = jnp.hstack((state_mean, ac_mean))
        # del_s = del_sa[:self.nS]
        # del_a = del_sa[self.nS:]
        sa_p_del_sa = sa + del_sa
        sa_m_del_sa = sa - del_sa
        
        # state_mean_p_del_s = state_mean + del_s
        # state_mean_m_del_s = state_mean - del_s
        
        # ac_mean_p_del_a = ac_mean + del_a
        # ac_mean_m_del_a = ac_mean - del_a
        
        # Currently just computing wrt state
        # Shape: (nS,)
        y = self.dynamics_fn_wrapper(state_mean, ac_mean, None)[0]
        
        # Shape: (nS+1, nS)
        y_p_del_s = jax.vmap(self.dynamics_fn_wrapper, in_axes=(0, 0, None))(sa_p_del_sa[:, :self.nS], sa_p_del_sa[:, self.nS:], None)[0]
        y_m_del_s = jax.vmap(self.dynamics_fn_wrapper, in_axes=(0, 0, None))(sa_m_del_sa[:, :self.nS], sa_m_del_sa[:, self.nS:], None)[0]
        
        sop = (y_p_del_s + y_m_del_s - 2*y).T/delta**2
        return sop[:, :self.nS], sop[:, self.nS:]

    # Computes the second order derivatives of a vector

    def hessian(self, fn, wrt):
        return jax.jacfwd(jax.jacrev(fn, argnums=wrt), argnums=wrt)

    def diag_hessian_of_transition(self, s, a, wrt):
        stacked_hessian = self.hessian(self.next_state_fn, wrt, has_aux=True)(s, a)
        return jax.numpy.diagonal(stacked_hessian, axis1=1, axis2=2)

    #########
    # Action Distribution initialization
    #########

    # actions ~ Uniform(0,1).
    # If saved, then replace the corresponding action in the sample
    # This has not been JIT intentionally
    def init_ac_mean(self, key):
        actions = jax.random.uniform(
            key, shape=(self.n_res, self.depth, self.nA))
        actions = actions.at[0, : self.depth -
                             1, :].set(self.saved_restart[1:, :])
        return actions

    # Uniform distribution variance : (1/12)(b-a)^2
    @partial(jax.jit, static_argnums=(0,))
    def init_ac_var(self, action_means, low_action=0, high_action=1):
        lower_limit = jnp.abs(action_means - low_action)
        higher_limit = jnp.abs(action_means - high_action)
        closer_limit = jnp.minimum(lower_limit, higher_limit)
        variance = jnp.square(2 * closer_limit) / 12
        return variance

    @partial(jax.jit, static_argnums=(0,))
    def transform_ac_mean(self, free_action_means):
        transformed_actions = self.ac_lb + self.multiplicative_factor * free_action_means
        return transformed_actions

    @partial(jax.jit, static_argnums=(0,))
    def transform_ac_var(self, free_action_variance):
        return jnp.square(self.multiplicative_factor) * free_action_variance

    #########
    # Distribution fns for reward
    #########

    def second_order_partials_for_reward(self, operands):
        s_mean, a_mean, ns_mean = operands
        sop_s = jnp.diag(jax.hessian(self.rewards_fn_wrapper, argnums=(0))(
            s_mean, a_mean, ns_mean, None))
        sop_a = jnp.diag(jax.hessian(self.rewards_fn_wrapper, argnums=(1))(
            s_mean, a_mean, ns_mean, None))
        sop_ns = jnp.diag(jax.hessian(self.rewards_fn_wrapper, argnums=(2))(
            s_mean, a_mean, ns_mean, None))
        # sop_s = {k1: v1[k1] for k1, v1 in hess_s.items()}
        # sop_a = {k1: v1[k1] for k1, v1 in hess_a.items()}
        # sop_ns = {k1: v1[k1] for k1, v1 in hess_ns.items()}
        return sop_s, sop_a, sop_ns

    def reward_mean(self, state_mean, state_var, ac_mean, ac_var, ns_mean, ns_var, key):
        return self.rewards_fn_wrapper(state_mean, ac_mean, ns_mean, key)

    def reward_comp(self, state_mean, state_var, ac_mean, ac_var, ns_mean, ns_var, key):
        reward_for_mean_tau = self.rewards_fn_wrapper(state_mean, ac_mean, ns_mean, key)
        (sop_s, sop_a, sop_ns) = self.second_order_partials_for_reward(
            (state_mean, ac_mean, ns_mean))

        return reward_for_mean_tau + 0.5*(jnp.multiply(sop_a, ac_var).sum(axis=0) + jnp.multiply(sop_s, state_var).sum(axis=0) + jnp.multiply(sop_ns, ns_var).sum(axis=0))
