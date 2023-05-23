import time
import numpy as np
import jax
from planners.disprod import ContinuousDisprod

def compute_avg_action_time(env, cfg_env, g_obs_keys, ga_keys):
    def _compute_avg_action_time(cfg, mode, depth):
        fn_start = time.time()
        cfg["depth"] = depth
        cfg["disprod"]["taylor_expansion_mode"] = mode
        agent = ContinuousDisprod(cfg, cfg_env)
        print(f"Computing avg time for {mode} and {depth}", flush=True)
        agent_key = jax.random.PRNGKey(cfg["seed"])
        prev_ac_seq, agent_key = agent.reset(agent_key)
        state = env.reset()
        times = []
        for step in range(10):
            start = time.time()
            # action selection:
            obs_array = np.array([state[i] for i in g_obs_keys])
            # replace the following line of code with your agent call
            ac_array, k_idx, prev_ac_seq, agent_key = agent.choose_action(obs_array, prev_ac_seq, agent_key)
            action = {ga_keys[idx]: float(ac_array[idx]) for idx in k_idx}
            finish = time.time()
            next_state, _, done, info = env.step(action)
            print(f"[Time: {finish-start}] [H-{mode}-{depth}] Action generated {action}", flush=True)
            times.append(finish-start)
            state = next_state
        del agent
        fn_end = time.time()
        print(f"[Time: {fn_end-fn_start}] [H-{mode}-{depth}] Completed heuristic scan {action}", flush=True)
        return (np.mean(times), (mode, depth))
    return _compute_avg_action_time