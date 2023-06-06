import jax
import jax.numpy as jnp

from functools import partial
from planners.utils import adam_with_clipping, adam_with_projection
from planners.helpers import ns_and_reward_partial
from utils.common_utils import load_method
import gym.spaces as spaces
from pyRDDLGymHelper.Core.Jax import JaxRDDLLogic, JaxRDDLBackpropPlanner


DISPROD_NOISE_VARS = ["disprod_eps_norm", "disprod_eps_uni"]

class ContinuousDisprod():
    def __init__(self, cfg, rddl_model, cfg_env={}):

        ga_keys = cfg_env['ga_keys']
        s_keys = cfg_env["s_keys"]
        a_keys = cfg_env["a_keys"]
        ns_keys = cfg_env["ns_keys"]
        s_gs_idx = cfg_env["s_gs_idx"]
        a_ga_idx = cfg_env["a_ga_idx"]
        self.n_output = cfg_env["n_concurrent_ac"]
        
        self.nA = cfg_env["nA"]
        self.nS = cfg_env["nS"]
        self.depth = cfg.get("depth")

        mode = cfg["mode"]
        self.n_res_lr = cfg[mode]["n_restarts"]
        self.max_grad_steps = cfg[mode]["max_grad_steps"]
        self.step_size = cfg[mode]["step_size"]
        self.step_size_var = cfg[mode]["step_size_var"]
        self.conv_thresh = cfg[mode]["convergance_threshold"]
            
        self.bool_s_idx = jnp.array(cfg_env['bool_s_idx'], dtype=jnp.int32)
        self.bool_ga_idx = jnp.array(cfg_env['bool_ga_idx'], dtype=jnp.int32)
        self.real_ga_idx = jnp.array(cfg_env['real_ga_idx'], dtype=jnp.int32)

        tnorm = getattr(JaxRDDLLogic, cfg['tnorm'])(**cfg['tnorm_kwargs'])
        fuzzy_logic = getattr(JaxRDDLLogic, cfg['logic'])(tnorm=tnorm, **cfg['logic_kwargs'])

        jax_compiled_model = JaxRDDLBackpropPlanner.JaxRDDLCompilerWithGrad(rddl=rddl_model, logic=fuzzy_logic)
        jax_compiled_model.compile()

        noop_ac = {}
        for k in jax_compiled_model.rddl.actions.keys():
            noop_ac.update(jax_compiled_model.rddl.ground_values(k, jax_compiled_model.init_values[k]))
        g_noop_ac = jnp.array([noop_ac[k] for k in ga_keys])

        ns_and_rew_fn = ns_and_reward_partial(jax_compiled_model, s_keys, a_keys, ns_keys, s_gs_idx, a_ga_idx)

        # Setup action bounds
        ac_bounds_user = cfg[mode].get("action_bounds", {})
        overwrite_ac_bounds = cfg["overwrite_ac_bounds"]
        self.ac_lb, self.ac_ub = compute_ac_bounds(cfg_env["action_space"], ga_keys, overwrite_ac_bounds, ac_bounds_user, cfg["posinf"], cfg["neginf"])
        # Multiplicative factor used to transform free_action variables to the legal range.
        self.scale_fac = self.ac_ub - self.ac_lb
        
        # Partial function to initialize action distribution
        noop_init = (g_noop_ac - self.ac_lb)/self.scale_fac
        self.ac_dist_init_fn = init_real_ac_dist(self.n_res_lr, self.depth, self.nA, noop_init, low_ac=0, high_ac=1, bool_ac_idx = self.bool_ga_idx)
        
        # Partial function to scale action distribution
        self.ac_dist_trans_fn = trans_ac_dist(self.ac_lb, self.scale_fac)

        if cfg[mode]["choose_action_mean"]:
            self.ac_selector = lambda m,v,key: m
        else:
            self.ac_selector = lambda m,v,key: m + jnp.sqrt(v) * jax.random.normal(key, shape=(self.nA,)) 

        # Support for normal, uniform, weibull and bernoulli.
        noise_dist = { "uni_mu"    : 0.5,
                        "uni_var"   : 1/12,
                        "norm_mu"   : 0,
                        "norm_var"  : 1}
        self.n_noise = 2
    

        # Setup transition and reward function
        if mode == "complete":
            fop_fn = fop_analytic(ns_and_rew_fn)
            ns_and_rew_concat_fn = ns_and_rew_concat(ns_and_rew_fn)
            if cfg[mode]["sop"] == "analytic":
                sop_fn = sop_analytic(ns_and_rew_concat_fn)
            else:
                sop_fn = sop_numerical(ns_and_rew_concat_fn, self.nS + self.n_noise, self.nA)
            dist_fn = dynamics_comp(ns_and_rew_fn, fop_fn, sop_fn, noise_dist, self.bool_s_idx, self.nS)
        elif mode == "no_var":
            dist_fn = dynamics_nv(ns_and_rew_fn, noise_dist)
        elif mode == "sampling":
            dist_fn = dynamics_sampling(ns_and_rew_fn, noise_dist)
        else:
            raise Exception(
                f"Unknown mode. Got {mode}")
            
        self._setup_q_fn(noise_dist, dist_fn)
        self._setup_projection(cfg, rddl_model, cfg_env)

        del jax_compiled_model
        del rddl_model

        
    def pre_warm(self, dummy_obs):
        # dummy_obs =  jnp.zeros(self.nS)
        dummy_key_1, dummy_key_2 = jax.random.split(jax.random.PRNGKey(0))
        dummy_ac_seq = jax.random.uniform(dummy_key_1, shape=(self.depth, self.nA))
        dummy_lrs_to_scan = jnp.ones((3, self.nA))
        _, _, _, _, grad_mean = self.choose_action(dummy_obs, dummy_ac_seq, dummy_key_2, dummy_lrs_to_scan)
        __lr = 1/jnp.mean(jnp.abs(grad_mean), axis=0)
        for i in range(self.nA):
            while __lr[i] > self.ac_ub[i]:
                __lr = __lr.at[i].set(__lr[i]/10)
        return jnp.array([__lr*10, __lr, __lr/10])

    def _setup_q_fn(self, noise_dist, dist_fn):
        rollout_fn = rollout_graph(dist_fn)
        q_fn = q(noise_dist, self.depth, rollout_fn)
        self.batch_q_fn = jax.vmap(q_fn, in_axes=(0, 0, 0, None), out_axes=(0, None))
        
        self.batch_grad_q_fn = jax.vmap(grad_q(q_fn), in_axes=(0, 0, 0, None), out_axes=(0, 0))

    def _setup_projection(self, cfg, rddl_model, cfg_env):
        # projection_fn is for a row of actions. vmap here works on the depth axis.
        projection_fn = load_method(cfg["projection_fn"])
        if cfg["env_name"] == "recsim":
            n_consumer = len(rddl_model.objects["consumer"])
            n_item = len(rddl_model.objects["item"]) 
            projection_fn = jax.vmap(partial(projection_fn, len(cfg_env["bool_ga_idx"]), n_consumer, n_item), in_axes=(0), out_axes=(0))
        else:
            projection_fn = jax.vmap(partial(projection_fn, len(cfg_env["bool_ga_idx"])), in_axes=(0), out_axes=(0))
        # projection_fn = cfg_env["projection_fn"]
        self.batch_projection = jax.vmap(projection_fn, in_axes=(0), out_axes=(0))

    def reset(self, key):
        key_1, key_2 = jax.random.split(key)
        ac_seq = jax.random.uniform(key_1, shape=(self.depth, self.nA))
        return ac_seq, key_2

    @partial(jax.jit, static_argnums=(0,))
    def choose_action(self, obs, prev_ac_seq, key, lrs_to_scan):
        ac_seq = prev_ac_seq

        lr_matrix = setup_lr_matrix(lrs_to_scan, self.n_res_lr, self.depth)
        n_res = lr_matrix.shape[0]

        # Create a vector of obs corresponding to n_restarts
        state = jnp.tile(obs, (n_res, 1)).astype('float32')

        # key: returned
        # subkey1: for action distribution initialization
        # subkey2: for random argmax
        # subkey3: for sampling action from action distribution
        # subkey4: for sampling noise in encapsulate mode
        key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 5)

        # Initialize the action distribution. Shape: (n_res, depth, nA)
        ac_mean, ac_var = self.ac_dist_init_fn(subkey1, ac_seq, 3)

        # Optimizer for continuous action variables - mean and variance
        opt_init_mean, opt_update_mean, get_params_mean = adam_with_clipping(self.step_size)
        opt_state_mean = opt_init_mean(ac_mean[:, :, self.real_ga_idx])

        opt_init_var, opt_update_var, get_params_var = adam_with_clipping(self.step_size_var)
        opt_state_var = opt_init_var(ac_var)

        # Optimizer for binary action variables - mean
        opt_init_bin, opt_update_bin, get_params_bin = adam_with_projection(self.step_size, proj_fn=self.batch_projection)
        opt_state_bin = opt_init_bin(ac_mean[:, :, self.bool_ga_idx])

        n_grad_steps = 0
        has_converged = False

        def _update_ac(val):
            """
            Update loop for all the restarts.
            """
            ac_mu_init, ac_var_init, n_grad_steps, has_converged, state, opt_state_mean, opt_state_var, opt_state_bin, grad_mean_stats, _key = val

            # Scale action means and variance from (0,1) to permissible action ranges
            scaled_ac_mu, scaled_ac_var = self.ac_dist_trans_fn(ac_mu_init, ac_var_init)

            # Compute Q-value function for all restarts
            reward, _subkey1 = self.batch_q_fn(state, scaled_ac_mu, scaled_ac_var, _key)

            # Compute gradients with respect to action means and action variance.
            grad_mean, grad_var = self.batch_grad_q_fn(state, scaled_ac_mu, scaled_ac_var, _key)

            # Update mean of real actions.
            opt_state_mean = opt_update_mean(n_grad_steps, -grad_mean[:, :, self.real_ga_idx], opt_state_mean, 0, 1)
            r_ac_mu_upd = get_params_mean(opt_state_mean)
            ac_mu_upd = ac_mu_init.at[:, :, self.real_ga_idx].set(r_ac_mu_upd)

            # Update variance of real actions.
            wiggle_room = jnp.minimum(ac_mu_upd - 0, 1 - ac_mu_upd)
            opt_state_var = opt_update_var(n_grad_steps, -grad_var, opt_state_var, 0, jnp.minimum(1/12, jnp.square(wiggle_room)/12))
            ac_var_upd = get_params_var(opt_state_var)

            # Update mean of boolean actions
            opt_state_bin = opt_update_bin(n_grad_steps, -grad_mean[:, :, self.bool_ga_idx], opt_state_bin)
            b_ac_mu_upd = get_params_bin(opt_state_bin)
            ac_mu_upd = ac_mu_upd.at[:, :, self.bool_ga_idx].set(b_ac_mu_upd)
        
            # Update variance of boolean actions
            bool_ac_var = ac_mu_upd[:, :, self.bool_ga_idx] * (1-ac_mu_upd[:, :, self.bool_ga_idx])
            ac_var_upd = ac_var_upd.at[:, :, self.bool_ga_idx].set(bool_ac_var)

            # Scale updated action means and variance from (0,1) to permissible action ranges
            scaled_ac_mu_upd, scaled_ac_var_upd = self.ac_dist_trans_fn(ac_mu_upd, ac_var_upd)

            # Compute updated reward
            updated_reward, _subkey2 = self.batch_q_fn(state, scaled_ac_mu_upd, scaled_ac_var_upd, _subkey1)

            # Reset the restarts in which updates led to a poor reward
            restarts_to_reset = jnp.where(updated_reward < reward, jnp.ones(n_res, dtype=jnp.int32), jnp.zeros(n_res, dtype=jnp.int32))
            mask = jnp.tile(restarts_to_reset, (self.depth, self.nA, 1)).transpose(2, 0, 1)
            ac_mu = ac_mu_init * mask + ac_mu_upd * (1-mask)
            ac_var = ac_var_init * mask + ac_var_upd * (1-mask)

            # Compute action mean and variance epsilon
            ac_mu_eps = ac_mu - ac_mu_init
            ac_var_eps = ac_var - ac_var_init

            # Check for convergence
            has_converged = jnp.logical_and(jnp.max(jnp.abs(ac_mu_eps)) < self.conv_thresh, jnp.max(jnp.abs(ac_var_eps)) < self.conv_thresh/10)

            grad_mean_stats = grad_mean_stats.at[n_grad_steps].set(jnp.mean(jnp.abs(grad_mean), axis=0))

            return ac_mu, ac_var, n_grad_steps + 1, has_converged, state, opt_state_mean, opt_state_var, opt_state_bin, grad_mean_stats, _subkey2

        def _check_conv(val):
            _, _, n_grad_steps, has_converged, _, _, _, _, _, _ = val
            return jnp.logical_and(n_grad_steps < self.max_grad_steps, jnp.logical_not(has_converged))

        # Iterate until max_grad_steps reached or both means and variance has not converged

        grad_mean_stats_ = jnp.zeros((self.max_grad_steps, self.depth, self.nA))
        init_val = (ac_mean, ac_var, n_grad_steps, has_converged, state, opt_state_mean, opt_state_var, opt_state_bin, grad_mean_stats_, subkey4)        
        ac_mean, ac_var, n_grad_steps, _, _, _, _,_, grad_mean_stats, subkey5 = jax.lax.while_loop(_check_conv, _update_ac, init_val)

        scaled_ac_mean, scaled_ac_var = self.ac_dist_trans_fn(ac_mean, ac_var)

        q_value, _ = self.batch_q_fn(state, scaled_ac_mean, scaled_ac_var, subkey5)

        best_restart = jnp.nanargmax(q_value)
        ac_seq = ac_mean[best_restart]

        ac = self.ac_selector(scaled_ac_mean[best_restart][0], scaled_ac_var[best_restart][0], subkey3)
        ac = jnp.clip(ac, self.ac_lb, self.ac_ub)
        
        _, k_idx = jax.lax.top_k(ac, self.n_output)
        
        return ac, k_idx, ac_seq, key, jnp.mean(jnp.abs(grad_mean_stats), axis=0)



# def random_argmax(key, x, pref_idx=0):
#     try:
#         options = jnp.where(x == jnp.nanmax(x))[0]
#         val = 0 if 0 in options else jax_random.choice(key, options)
#     except:
#         val = jax_random.choice(key, jnp.arange(len(x)))
#         print(f"All restarts where NaNs. Randomly choosing {val}.")
#     finally:
#         return val


#########
# Action Distribution initialization and transformation
#########

# actions ~ Uniform(0,1).
def init_real_ac_dist(n_res, depth, nA, noop_init, low_ac, high_ac, bool_ac_idx):
    def _init_real_ac_dist(key, ac_seq, n_repeats):
        ac_mean = jax.random.uniform(key, shape=(n_res, depth, nA))
        ac_mean = ac_mean.at[0, : depth -1, :].set(ac_seq[1:, :])

        # Have two extremes as a part of the search space
        ac_mean = ac_mean.at[1, :, :].set(jnp.zeros((depth, nA)))
        ac_mean = ac_mean.at[2, :, :].set(jnp.ones((depth, nA)))
        # Set this to translate to no-op action
        ac_mean = ac_mean.at[3, :, :].set(noop_init)

        lower_limit = jnp.abs(ac_mean - low_ac)
        higher_limit = jnp.abs(ac_mean - high_ac)
        closer_limit = jnp.minimum(lower_limit, higher_limit)
        ac_var = jnp.square(2 * closer_limit) / 12
        ac_var = ac_var.at[bool_ac_idx].set(ac_mean[bool_ac_idx] * (1-ac_mean[bool_ac_idx]))

        # Repeat along the restart dimension but not along depth and nA
        ac_mean = jnp.tile(ac_mean, (n_repeats, 1, 1))
        ac_var = jnp.tile(ac_var, (n_repeats, 1, 1))
        return ac_mean, ac_var
    return _init_real_ac_dist

def trans_ac_dist(ac_lb, scale_fac):
    def _trans_ac_dict(ac_mean, ac_var):
        scaled_ac_mean = ac_lb + scale_fac * ac_mean
        scaled_ac_var = jnp.square(scale_fac) * ac_var
        return scaled_ac_mean, scaled_ac_var
    return _trans_ac_dict


# https://arxiv.org/pdf/1309.1541.pdf
def projection_fn(nA):
    def _projection_fn(ac):
        ac_sort = jnp.sort(ac)[::-1]
        ac_sort_cumsum = jnp.cumsum(ac_sort)
        rho_candidates = ac_sort + (1 - ac_sort_cumsum)/jnp.arange(1, nA+1)
        mask = jnp.where(rho_candidates > 0, jnp.arange(nA, dtype=jnp.int32), -jnp.ones(nA, dtype=jnp.int32))
        rho = jnp.max(mask)
        contrib = (1 - ac_sort_cumsum[rho])/(rho + 1)
        return jax.nn.relu(ac + contrib)
    return jax.vmap(_projection_fn, in_axes=0, out_axes=0)

#####################################
# Q-function computation graph
#################################

def rollout_graph(dynamics_dist_fn):
    def _rollout_graph(d, params):
        agg_reward, s_mu, s_var, a_mu, a_var, key = params
        ns_mu, ns_var, reward, key = dynamics_dist_fn(s_mu, s_var, a_mu[d, :], a_var[d, :], key)            
        return agg_reward+reward, ns_mu, ns_var, a_mu, a_var, key
    return _rollout_graph

def q(noise_dist, depth, rollout_fn):
    def _q(s, a_mu, a_var, key):
        """
        Compute the Q-function for a single restart
        """
        # augment state by adding variable for noise    
        s_mu = jnp.concatenate([s, jnp.array([noise_dist["norm_mu"], noise_dist["uni_mu"]])], axis=0)
        s_var = jnp.concatenate([s * 0, jnp.array([noise_dist["norm_var"], noise_dist["uni_var"]])], axis=0)

        init_rew = jnp.array([0.0])
        init_params = (init_rew, s_mu, s_var, a_mu, a_var, key)
        agg_rew, _, _, _, _, key = jax.lax.fori_loop( 0, depth, rollout_fn, init_params)
        return agg_rew.sum(), key
    return _q
    
def grad_q(q):    
    def _grad_q(s, ac_mu, ac_var, key):
        """
        Compute the gradient of Q-function for a single restart
        """
        grads = jax.grad(q, argnums=(1,2), has_aux=True)(s, ac_mu, ac_var, key)[0]
        return grads[0], grads[1]
    return _grad_q

#####################################
# Dynamics Distribution Fn
####################################

# No variance mode
def dynamics_nv(ns_and_rew_fn, noise_dist):
    def _dynamics_nv(s_mu, s_var, a_mu, a_var, key):
        ns,reward = ns_and_rew_fn(s_mu, a_mu, None)
        
        ns_mu = jnp.concatenate([ns, jnp.array([noise_dist["norm_mu"], noise_dist["uni_mu"]])], axis=0)
        ns_var = jnp.concatenate([jnp.zeros_like(ns), jnp.array([noise_dist["norm_var"], noise_dist["uni_var"]])], axis=0)

        return ns_mu, ns_var, reward, key
    return _dynamics_nv

# Complete Mode
def dynamics_comp(ns_and_rew_fn, fop_fn, sop_fn, noise_dist, bin_idx, nS):
    def _dynamics_comp(s_mu, s_var, a_mu, a_var, key):
        ns, reward = ns_and_rew_fn(s_mu, a_mu, None)
        
        fop_wrt_s, fop_wrt_ac = fop_fn(s_mu, a_mu)

        # SOP is also computed wrt reward
        sop_wrt_s, sop_wrt_ac = sop_fn(s_mu, a_mu)

        # Expected reward
        reward_mu = reward + 0.5*(jnp.multiply(sop_wrt_ac[nS:nS+1], a_var).sum(axis=1) + jnp.multiply(sop_wrt_s[nS:nS+1], s_var).sum(axis=1))

        # Taylor's expansion
        ns_mu = ns + 0.5*(jnp.multiply(sop_wrt_ac[0:nS], a_var).sum(axis=1) + jnp.multiply(sop_wrt_s[0:nS], s_var).sum(axis=1))
        ns_var = jnp.multiply(jnp.square(fop_wrt_ac), a_var).sum(axis=1) + jnp.multiply(jnp.square(fop_wrt_s), s_var).sum(axis=1)

        mu_bin = jnp.clip(ns_mu[bin_idx], 0, 1)
        ns_mu = ns_mu.at[bin_idx].set(mu_bin)

        var_bin = mu_bin * (1-mu_bin)
        ns_var = ns_var.at[bin_idx].set(var_bin)

        ns_mu = jnp.concatenate([ns_mu, jnp.array([noise_dist["norm_mu"], noise_dist["uni_mu"]])], axis=0)
        ns_var = jnp.concatenate([ns_var, jnp.array([noise_dist["norm_var"], noise_dist["uni_var"]])], axis=0)

        return ns_mu, ns_var, reward_mu, key
    return _dynamics_comp

# Sampling Mode
def dynamics_sampling(ns_and_rew_fn, noise_dist):
    def _dynamics_sampling(s_mu, s_var, a_mu, a_var, key):
        
        subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 4)
        eps_norm = jax.random.normal(subkey1)
        eps_uni = jax.random.uniform(subkey2)
        s_mu = s_mu.at[-2].set(eps_norm)
        s_mu = s_mu.at[-1].set(eps_uni)
        
        ns, reward = ns_and_rew_fn(s_mu, a_mu, subkey3)
        
        ns_mu = jnp.concatenate([ns, jnp.array([noise_dist["norm_mu"], noise_dist["uni_mu"]])], axis=0)
        ns_var = jnp.concatenate([jnp.zeros_like(ns), jnp.array([noise_dist["norm_var"], noise_dist["uni_var"]])], axis=0)

        return ns_mu, ns_var, reward, subkey4
    return _dynamics_sampling

#################################
# Wrapper for computing second order partials
###################################
def ns_and_rew_concat(ns_and_rew_fn):
    def _ns_and_rew_concat(s_mu, a_mu, rng_key):
        ns, rew = ns_and_rew_fn(s_mu, a_mu, rng_key)
        return jnp.hstack((ns, rew))
    return _ns_and_rew_concat

#########################################
# Functions for computing partials - Analytic
###########################################

# Dont need FOP for reward, so this uses ns_and_rew_fn
def fop_analytic(ns_and_rew_fn):
    def _fop_analytic(s_mu, a_mu):
        return jax.jacfwd(ns_and_rew_fn, has_aux=True, argnums=(0, 1))(s_mu, a_mu, None)[0]
    return _fop_analytic

# Need SOP for reward, so this uses ns_and_rew_concat
def sop_analytic(ns_and_rew_concat):
    def _sop_analytic(s_mu, a_mu):
        def _diag_hessian(wrt):
            hess = jax.hessian(ns_and_rew_concat, wrt)(s_mu, a_mu, None)
            return jax.numpy.diagonal(hess, axis1=1, axis2=2)
        # TODO: Compute in one call
        sop_wrt_s = _diag_hessian(0)
        sop_wrt_ac = _diag_hessian(1)
        return sop_wrt_s, sop_wrt_ac
    return _sop_analytic

#########################################
# Functions for computing partials - Numerical
###########################################

# Need SOP for reward, so this uses ns_and_rew_concat
def sop_numerical(ns_and_rew_concat, nS, nA):
    def _fop_partial(s_mu, a_mu):
        return jax.jacfwd(ns_and_rew_concat, argnums=(0, 1))(s_mu, a_mu, None)
    
    def _sop_numerical(s_mu, a_mu):
        
        # For del_s, concat extra zero rows at the bottom.
        # For del_a, concat extra zero rows at the top.
        delta = 1e-2
        # del_s = jnp.vstack([delta * jnp.eye(self.nS), jnp.zeros(self.nA, self.nS + self.nA)])
        # del_a = jnp.vstack([jnp.zeros(self.nA, self.nS + self.nA), delta * jnp.eye(self.nA)])
        
        del_sa = delta * jnp.eye(nS + nA)
        sa = jnp.hstack((s_mu, a_mu))
        # del_s = del_sa[:self.nS]
        # del_a = del_sa[self.nS:]
        sa_p_del_sa = sa + del_sa
        # sa_m_del_sa = sa - del_sa
        
        # state_mean_p_del_s = state_mean + del_s
        # state_mean_m_del_s = state_mean - del_s
        
        # ac_mean_p_del_a = ac_mean + del_a
        # ac_mean_m_del_a = ac_mean - del_a
        
        # Currently just computing wrt state
        # Shape: (nS,)
        y_s, y_a = _fop_partial(s_mu, a_mu)
        
        y_p_s, y_p_a = jax.vmap(_fop_partial, in_axes=(0, 0), out_axes=0)(sa_p_del_sa[:, : nS], sa_p_del_sa[:, nS:])
        # Shape: (nS+1, nS)
        # y_p_del_s = jax.vmap(ns_and_rew_concat, in_axes=(0, 0, None))(sa_p_del_sa[:, : nS], sa_p_del_sa[:, nS:], None)
        # y_m_del_s = jax.vmap(ns_and_rew_concat, in_axes=(0, 0, None))(sa_m_del_sa[:, : nS], sa_m_del_sa[:, nS:], None)
        
        sop_s = (jnp.diagonal(y_p_s, axis1=0, axis2=2) - y_s)/delta
        sop_a = (jnp.diagonal(y_p_a, axis1=0, axis2=2) - y_a)/delta
        return sop_s, sop_a
    return _sop_numerical

def compute_ac_bounds(ac_space, a_keys, overwrite_ac_bounds, ac_bounds_user, posinf=100, neginf=-100):
    ac_lb, ac_ub = [], []
    for a in a_keys:
        ac_obj = ac_space[a]
        if overwrite_ac_bounds:
            _bounds = ac_bounds_user[a.split("__")[0].strip()]
            ac_lb.append(_bounds[0])
            ac_ub.append(_bounds[1])
        else:
            if type(ac_obj) == spaces.discrete.Discrete:
                ac_lb.append(ac_obj.start)
                ac_ub.append(ac_obj.start + ac_obj.n - 1)
            else:
                ac_lb.append(ac_obj.low[0])
                ac_ub.append(ac_obj.high[0])
    return jnp.nan_to_num(jnp.array(ac_lb, dtype=jnp.float32), posinf=posinf, neginf=neginf), jnp.nan_to_num(jnp.array(ac_ub, dtype=jnp.float32), posinf=posinf, neginf=neginf)

def setup_lr_matrix(lr_arr, n_res_lr, depth):
    lr_matrix = jnp.vstack([jnp.tile(lr_arr[0], ((n_res_lr, depth, 1))),
                            jnp.tile(lr_arr[1], ((n_res_lr, depth, 1))),
                            jnp.tile(lr_arr[2], ((n_res_lr, depth, 1)))])
    return lr_matrix