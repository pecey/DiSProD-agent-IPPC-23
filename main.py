import sys
import signal
import time
# sys.path.append('/home/test/pyRDDLGym')
import numpy as np
from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Policies.Agents import NoOpAgent
import copy
from functools import partial

# for JAX backend:
# from pyRDDLGym.Core.Jax.JaxRDDLSimulator import JaxRDDLSimulator

############################################################
# IMPORT THE AGENT AND OTHER DEPENDENCIES OF YOUR SOLUTION #
from utils import helpers, heuristics
from functools import partial
from planners.disprod import ContinuousDisprod
import jax
from utils.common_utils import prepare_config
import os
import multiprocessing as mp
from concurrent.futures import as_completed, ProcessPoolExecutor      

DISPROD_NOISE_VARS = ["disprod_eps_norm", "disprod_eps_uni"]
############################################################

HOUR = 3600
def signal_handler(signum, frame):
    raise Exception("Timed out!")


# MAIN INTERACTION LOOP #
def main(env, inst, method_name=None, episodes=1):
    print(f'preparing to launch instance {inst} of domain {env}...')

    # get the environment info
    EnvInfo = ExampleManager.GetEnvInfo(env)

    # set up the environment class, choose instance 0 because every example has at least one example instance
    log = False if method_name is None else True
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(),
                            instance=EnvInfo.get_instance(inst),
                            enforce_action_constraints=False,
                            debug=True,
                            log=log,
                            simlogname=method_name)
                            # backend=JaxRDDLSimulator)

    # initialization of the method init timer
    budget = myEnv.Budget
    init_budget = HOUR
    init_timed_out = False
    signal.signal(signal.SIGALRM, signal_handler)

    # default noop agent, do not change
    defaultAgent = NoOpAgent(action_space=myEnv.action_space,
                        num_actions=myEnv.numConcurrentActions)

    signal.setitimer(signal.ITIMER_REAL, init_budget)
    start = time.time()
    try:
        ################################################################
        # Initialize your agent here:

        current_dir = os.path.dirname(os.path.abspath(__file__))
        cfg = prepare_config("_".join(env.lower().split()), f"{current_dir}/config")

        checkpoint = time.time()
        print(f"[Time: {checkpoint-start}] Loading config from path {current_dir}/config")

        # Don't reparameterize the RDDL expressions if planner uses sampling
        reparam_rddl = True if cfg["disprod"]["taylor_expansion_mode"] in ["complete", "no_var"] else False
        domain_path = EnvInfo.get_domain()
        instance_path = EnvInfo.get_instance(inst)
        rddl_model = helpers.gen_model(domain_path, instance_path, reparam_rddl)

        checkpoint = time.time()
        print(f"[Time: {checkpoint-start}] Reparam RDDL set to {reparam_rddl}")

        g_obs_keys = [key for key in rddl_model.groundstates().keys() if key not in DISPROD_NOISE_VARS]

        s_keys, a_keys, ga_keys, ns_keys, bool_s_idx, bool_ga_idx, real_ga_idx = helpers.prepare_rddl_compilations(rddl_model)

        init_subs = myEnv.sampler.subs

        # obs_keys = state_keys - noise_keys. g_obs_keys = grounded version of obs_keys
        obs_keys = [key for key in s_keys if key not in DISPROD_NOISE_VARS]
        
        # Map state/action to the indices capturing the grounded states/action in the transition function input vector
        s_gs_idx = helpers.prepare_index_mapping(obs_keys, rddl_model.grounded_names, init_subs, noise_vars=True)
        a_ga_idx = helpers.prepare_index_mapping(a_keys, rddl_model.grounded_names, init_subs, noise_vars=False)

        cfg_env = {}
        cfg_env["s_keys"] = obs_keys + DISPROD_NOISE_VARS
        cfg_env["a_keys"] = a_keys
        cfg_env["ns_keys"] = ns_keys
        cfg_env['ga_keys'] = ga_keys
        cfg_env['bool_s_idx'] = bool_s_idx
        cfg_env['bool_ga_idx'] = bool_ga_idx
        cfg_env['real_ga_idx'] = real_ga_idx
        cfg_env["action_space"] = myEnv.action_space
        cfg_env["n_concurrent_ac"] = myEnv.numConcurrentActions
        cfg_env["nA"] = len(myEnv.action_space)
        cfg_env["nS"] = len(myEnv.observation_space)
        cfg_env["s_gs_idx"] = s_gs_idx
        cfg_env["a_ga_idx"] = a_ga_idx

        
        checkpoint=time.time()

        # Setup default agent.
        agent = ContinuousDisprod(cfg, rddl_model, cfg_env)
        agent_key = jax.random.PRNGKey(cfg["seed"])
        prev_ac_seq, agent_key = agent.reset(agent_key)
        checkpoint = time.time()
        print(f"[Time: {checkpoint-start}] Basic agent initialized.")

        # Perform heuristic scans 

        #################################################################
        # H1: Compute the average time taken per mode
        ##################################################################
        combs = [("no_var", cfg["depth"]), ("sampling", cfg["depth"])]
        scan_res = []
        heuristic_fn = partial(heuristics.compute_avg_action_time, domain_path, instance_path, rddl_model, cfg_env, g_obs_keys, ga_keys)

        # JAX doesn't fork with fork context which is default for Linux. Start a spawn context explicitly.
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(mp_context=context,max_workers=4) as executor:
            jobs = [executor.submit(heuristic_fn, copy.deepcopy(cfg), mode, depth) for mode, depth in combs]

            for job in as_completed(jobs):
                result = job.result()
                print(result)
                scan_res.append(result)

        scan_res = sorted(scan_res, key=lambda x: (x[0], x[1]))
        print(scan_res)

        ##################################################################
        # H2: Search across different LRs
        #################################################################

        combs = [("no_var", 0.001), ("sampling", 0.1)]
        scan_res = []
        heuristic_fn = partial(heuristics.compute_score_stats, domain_path, instance_path, rddl_model, cfg_env, g_obs_keys, ga_keys, n_episodes=2)

        # JAX doesn't fork with fork context which is default for Linux. Start a spawn context explicitly.
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(mp_context=context,max_workers=4) as executor:
            jobs = [executor.submit(heuristic_fn, copy.deepcopy(cfg), mode, lr) for mode, lr in combs]

            for job in as_completed(jobs):
                result = job.result()
                print(result)
                scan_res.append(result)

        scan_res = sorted(scan_res, key=lambda x: (x[0], x[1]))
        print(scan_res)
        ################################################################
    except:
        finish = time.time()
        print('Timed out! (', finish - start, ' seconds)')
        print('This domain will continue exclusively with default actions!')
        init_timed_out = True

    # signal.signal(signal.SIGALRM, signal_handler)

    # for episode in range(episodes):
    #     total_reward = 0
    #     state = myEnv.reset()
    #     # timed_out = False if init_timed_out==False else True
    #     timed_out = False
    #     elapsed = budget
    #     start = 0
    #     for step in range(myEnv.horizon):

    #         # action selection:
    #         if not timed_out:
    #             signal.setitimer(signal.ITIMER_REAL, elapsed)
    #             start = time.time()
    #             try:
    #                 #################################################################
    #                 # replace the following line of code with your agent call
    #                 # action = agent.sample_action()
    #                 obs_array = np.array([state[i] for i in g_obs_keys])
    #                 # replace the following line of code with your agent call
    #                 ac_array, k_idx, prev_ac_seq, agent_key = agent.choose_action(obs_array, prev_ac_seq, agent_key)
    #                 action = {ga_keys[idx]: float(ac_array[idx]) for idx in k_idx}


    #                 #################################################################
    #                 finish = time.time()
    #                 print(f"[Time: {finish-start}] Action generated {action}")
    #             except:
    #                 finish = time.time()
    #                 print('Timed out! (', finish-start, ' seconds)')
    #                 print('This episode will continue with default actions!')
    #                 action = defaultAgent.sample_action()
    #                 timed_out = True
    #                 elapsed = 0
    #             if not timed_out:
    #                 elapsed = elapsed - (finish-start)
    #         else:
    #             action = defaultAgent.sample_action()

    #         next_state, reward, done, info = myEnv.step(action)
    #         total_reward += reward

    #         print()
    #         print(f'step       = {step}')
    #         print(f'state      = {state}')
    #         print(f'action     = {action}')
    #         print(f'next state = {next_state}')
    #         print(f'reward     = {reward}')

    #         state = next_state

    #         if done:
    #             break

    #     print(f'episode {episode+1} ended with reward {total_reward} after {budget-elapsed} seconds')

    myEnv.close()

    ########################################
    # CLEAN UP ANY RESOURCES YOU HAVE USED #


    ########################################


# Command line interface, DO NOT CHANGE
if __name__ == "__main__":
    # args = sys.argv
    # print(args)
    # method_name = None
    # episodes = 1
    # if len(args) == 2:
    #     if args[0] == '-h':
    #         print('python GymExample.py <domain> <instance> <method name> <num episodes>')
    # if len(args) < 3:
    #     env, inst = 'HVAC', '1'
    # elif len(args) < 4:
    #     env, inst = args[1:3]
    # elif len(args) < 5:
    #     env, inst, method_name = args[1:4]
    # else:
    #     env, inst, method_name, episodes = args[1:5]
    #     try:
    #         episodes = int(episodes)
    #     except:
    #         raise ValueError("episode must be an integer value argument, received: " + episodes)
    env="HVAC"
    inst=0
    method_name="disprod"
    episodes=1
    main(env, inst, method_name, episodes)

