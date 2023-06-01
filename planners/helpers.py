import jax.numpy as jnp
from pyRDDLGymHelper.Core.Jax import JaxRDDLBackpropPlanner

def ns_and_reward_partial(rddl_model, s_keys, a_keys, ns_keys, s_gs_idx, a_ga_idx):
    jax_compiled_model = JaxRDDLBackpropPlanner.JaxRDDLCompilerWithGrad(rddl=rddl_model)
    jax_compiled_model.compile()

    reward_fn = jax_compiled_model.reward
    cpfs = jax_compiled_model.cpfs
    extra_params = jax_compiled_model.model_params
    levels = [_v for v in jax_compiled_model.levels.values() for _v in v]
    const_dict = {k:jax_compiled_model.init_values[k] for k in jax_compiled_model.rddl.nonfluents.keys()}

    del rddl_model
    del jax_compiled_model

    def _ns_and_reward(state, action, rng_key):
        """
        s_keys, a_keys: not grounded 
        gs_keys, ga_keys: grounded
        grounded_names: map s_keys -> gs_keys, a_keys -> ga_keys
        state, action: grounded
        """

        # s_gs_idx and s_ga_idx have 4 values for each key.
        # idx 0 and idx 1 are the min and max indexes for the key
        # idx 3 is the desired shape.
        state_dict = {k: jnp.array(state[s_gs_idx[k][0] : s_gs_idx[k][1]]).reshape(s_gs_idx[k][3]) for k in s_keys}
        action_dict = {k: jnp.array(action[a_ga_idx[k][0] : a_ga_idx[k][1]]).reshape(a_ga_idx[k][3]) for k in a_keys}

        # subs should be not grounded.
        subs = {**state_dict, **action_dict, **const_dict}

        for level in levels:
            expr = cpfs[level]
            subs[level], _, _ = expr(subs, extra_params, rng_key)
            
        reward, _, _ = reward_fn(subs, extra_params, rng_key)

        # flatten is required in cases like RecSim where state variables are 2D
        return jnp.hstack([subs[k].flatten() for k in ns_keys]), reward
    return _ns_and_reward