import time
import numpy as np
import jax
from planners.disprod import ContinuousDisprod
from pyRDDLGym import RDDLEnv

def compute_avg_action_time(domain, instance, rddl_model, cfg_env, g_obs_keys, ga_keys, cfg, mode, depth):
    env = RDDLEnv.RDDLEnv(domain=domain,
                            instance=instance,
                            enforce_action_constraints=False,
                            debug=True)
    cfg["depth"] = depth
    cfg["mode"] = mode
    agent = ContinuousDisprod(cfg, rddl_model, cfg_env)
    agent_key = jax.random.PRNGKey(cfg["seed"])
    prev_ac_seq, agent_key = agent.reset(agent_key)
    state = env.reset()
    times = []
    for step in range(10):
        ac_start = time.time()
        # action selection:
        obs_array = np.array([state[i] for i in g_obs_keys])
        # replace the following line of code with your agent call
        ac_array, k_idx, prev_ac_seq, agent_key = agent.choose_action(obs_array, prev_ac_seq, agent_key)
        action = {ga_keys[idx]: float(ac_array[idx]) for idx in k_idx}
        ac_end = time.time()
        next_state, _, done, info = env.step(action)
        times.append(ac_end-ac_start)
        state = next_state
    del agent
    return (np.mean(times), np.std(times), (mode, depth))


def compute_score_stats(domain, instance, rddl_model, cfg_env, g_obs_keys, ga_keys, cfg, mode, lr, n_episodes=10):
    env = RDDLEnv.RDDLEnv(domain=domain,
                            instance=instance,
                            enforce_action_constraints=False,
                            debug=True)
    cfg[mode]["step_size"] = lr
    cfg[mode]["step_size_var"] = lr/10
    cfg["mode"] = mode
    agent = ContinuousDisprod(cfg, rddl_model, cfg_env)
    agent_key = jax.random.PRNGKey(cfg["seed"])
    prev_ac_seq, agent_key = agent.reset(agent_key)
    scores = []
    for i in range(n_episodes):
        state = env.reset()
        done = False
        agg_reward = 0
        while not done:
            # action selection:
            obs_array = np.array([state[i] for i in g_obs_keys])
            # replace the following line of code with your agent call
            ac_array, k_idx, prev_ac_seq, agent_key = agent.choose_action(obs_array, prev_ac_seq, agent_key)
            action = {ga_keys[idx]: float(ac_array[idx]) for idx in k_idx}
            next_state, reward, done, info = env.step(action)
            state = next_state
            agg_reward += reward
        scores.append(agg_reward)
    del agent
    return (np.mean(scores), np.std(scores), (mode, lr, lr/10))