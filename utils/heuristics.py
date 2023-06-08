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
    env.close()
    return (np.mean(times), np.std(times), (mode, depth))

COMPUTE_SCORE_STATS_TIME = 1200
def compute_score_stats(domain, instance, g_obs_keys, ga_keys, ac_dict_fn, cfg, mode, rddl_model, cfg_env, s_weight, n_episodes=10):
    start = time.time()
    env = RDDLEnv.RDDLEnv(domain=domain,
                            instance=instance,
                            enforce_action_constraints=False,
                            debug=True)
    cfg["mode"] = mode
    cfg["logic_kwargs"]["weight"] = s_weight
    agent = ContinuousDisprod(cfg, rddl_model, cfg_env)
    dummy_state = env.reset()
    dummy_obs = np.array([dummy_state[i] for i in g_obs_keys])
    lrs_to_scan = agent.pre_warm(dummy_obs)
    if cfg[mode]["overwrite_lrs"]:
        lrs_to_scan = cfg[mode]["lrs_to_scan"]
    agent_key = jax.random.PRNGKey(cfg["seed"])
    prev_ac_seq, agent_key = agent.reset(agent_key)
    scores = []
    end = time.time()
    # Evaluate the agent for one episode and compute how many episodes can we afford
    time_left = COMPUTE_SCORE_STATS_TIME - (end - start)
    agg_reward, ep_time, prev_ac_seq, agent_key = eval_episode(g_obs_keys, ga_keys, env, agent, agent_key, prev_ac_seq, lrs_to_scan, ac_dict_fn)
    scores.append(agg_reward)
    n_episodes = min(10, int(time_left/ep_time))
    
    # Evaluate the agent of the remaining number of episodes
    for i in range(n_episodes):
        agg_reward, _, prev_ac_seq, agent_key = eval_episode(g_obs_keys, ga_keys, env, agent, agent_key, prev_ac_seq, lrs_to_scan, ac_dict_fn)
        scores.append(agg_reward)
    del agent
    env.close()
    return np.mean(scores) - np.std(scores), mode, s_weight

def eval_episode(g_obs_keys, ga_keys, env, agent, agent_key, prev_ac_seq, lrs_to_scan, ac_dict_fn):
    start = time.time()
    state = env.reset()
    done = False
    agg_reward = 0
    while not done:
        obs_array = np.array([state[i] for i in g_obs_keys])
        ac_array, k_idx, prev_ac_seq, agent_key, _ = agent.choose_action(obs_array, prev_ac_seq, agent_key, lrs_to_scan)
        action = ac_dict_fn(ac_array, k_idx)
        next_state, reward, done, info = env.step(action)
        state = next_state
        agg_reward += reward
    end = time.time()
    return agg_reward, end-start, prev_ac_seq, agent_key