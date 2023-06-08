import time
import numpy as np
import jax
from planners.disprod import ContinuousDisprod
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Policies.Agents import NoOpAgent


def compute_score_stats(domain, instance, g_obs_keys, ga_keys, ac_dict_fn, cfg, mode, rddl_model, cfg_env, s_weight,  depth, n_episodes=10, time = 1200):
    start = time.time()
    env = RDDLEnv.RDDLEnv(domain=domain,
                            instance=instance,
                            enforce_action_constraints=False,
                            debug=True)
    cfg["mode"] = mode
    cfg["logic_kwargs"]["weight"] = s_weight
    cfg["depth"] = depth
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
    time_left = time - (end - start)
    agg_reward, ep_time, prev_ac_seq, agent_key = eval_episode(g_obs_keys, ga_keys, env, agent, agent_key, prev_ac_seq, lrs_to_scan, ac_dict_fn)
    scores.append(agg_reward)
    n_episodes = min(10, int(time_left/ep_time))
    
    # Evaluate the agent of the remaining number of episodes
    for i in range(n_episodes):
        agg_reward, _, prev_ac_seq, agent_key = eval_episode(g_obs_keys, ga_keys, env, agent, agent_key, prev_ac_seq, lrs_to_scan, ac_dict_fn)
        scores.append(agg_reward)
    del agent
    env.close()
    return np.mean(scores) - np.std(scores), mode, s_weight, depth

def eval_episode(g_obs_keys, ga_keys, env, agent, agent_key, prev_ac_seq, lrs_to_scan, ac_dict_fn):
    budget = 240
    start = time.time()
    state = env.reset()
    done = False
    agg_reward = 0
    while not done and budget > 0:
        start_ac = time.time()
        obs_array = np.array([state[i] for i in g_obs_keys])
        ac_array, k_idx, prev_ac_seq, agent_key, _ = agent.choose_action(obs_array, prev_ac_seq, agent_key, lrs_to_scan)
        action = ac_dict_fn(ac_array, k_idx)
        end_ac = time.time()
        next_state, reward, done, info = env.step(action)
        state = next_state
        agg_reward += reward
        budget = budget - (end_ac - start_ac)
    if budget <= 0 and not done:
        defaultAgent = NoOpAgent(action_space=env.action_space,
                        num_actions=env.numConcurrentActions)
        while not done:
            action = defaultAgent.sample_action()
            next_state, reward, done, info = env.step(action)
            agg_reward += reward
    end = time.time()
    return agg_reward, end-start, prev_ac_seq, agent_key