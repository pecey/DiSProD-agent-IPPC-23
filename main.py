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
from gym.spaces import Dict, Box

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
        print(f"[Time left: {init_budget - (checkpoint-start)}] Loading config from path {current_dir}/config")

        # Don't reparameterize the RDDL expressions if planner uses sampling
        domain_path = EnvInfo.get_domain()
        instance_path = EnvInfo.get_instance(inst)
        
        # Setup rddl_model for sampling mode, and reparam_rddl_model for NV and complete mode
        rddl_model = helpers.gen_model(domain_path, instance_path, False)
        reparam_rddl_model = helpers.gen_model(domain_path, instance_path, True)

        # For large RecSim instances and MarsRover       
        fallback = True if len(myEnv.action_space) > 250 else False
        heuristic_scan = False if fallback or cfg["skip_search"] else True
        
        
        if fallback and cfg["env_name"] == "recsim":
            g_obs_keys, ac_dict_fn, cfg_env = helpers.prepare_cfg_env_fallback_recsim(myEnv, cfg, rddl_model)
            del reparam_rddl_model
        else:
        
            # Setup cfg_env for sampling mode and reparam_cfg_env for NV and complete mode
            # TODO: Check if reparam_obs and reparam_a keys are different than normal?
            g_obs_keys, ga_keys, ac_dict_fn, cfg_env = helpers.prepare_cfg_env(env, myEnv, rddl_model, cfg)
            _, _, _, reparam_cfg_env = helpers.prepare_cfg_env(env, myEnv, reparam_rddl_model, cfg)
             
        # Get a dummy obs
        dummy_state = myEnv.reset()
        dummy_obs = np.array([dummy_state[i] for i in g_obs_keys])
        # Setup default agent depending on the default mode
        agent_setup_start = time.time()
        if cfg["mode"] == "sampling":
            agent = ContinuousDisprod(cfg, rddl_model, cfg_env)
            lrs_to_scan = agent.pre_warm(dummy_obs)
        else:
            agent = ContinuousDisprod(cfg, reparam_rddl_model, reparam_cfg_env)
            lrs_to_scan = agent.pre_warm(dummy_obs)
            
        agent_key = jax.random.PRNGKey(cfg["seed"])
        prev_ac_seq, agent_key = agent.reset(agent_key)
        if cfg[cfg["mode"]]["overwrite_lrs"]:
            lrs_to_scan = cfg[cfg["mode"]]["lrs_to_scan"]
        
        agent_setup_end = time.time()

        time_required_for_agent_setup = agent_setup_end-agent_setup_start
        print(f"[Time left: {init_budget - (agent_setup_end - start)}] Basic agent initialized. Time taken: {time_required_for_agent_setup}")

        # Perform heuristic scans       
        ##################################################################
        # H1: Search across different modes and see which is better
        ##################################################################
        if heuristic_scan:
            combs = []
            for mode in ["no_var", "sampling"]:
                for s_wt in [3, 5]:
                    for depth in [cfg["depth"], cfg["depth"]/2]:
                        if mode == "no_var":
                            model = reparam_rddl_model
                            _cfg_env = reparam_cfg_env
                        else:
                            model = rddl_model
                            _cfg_env = cfg_env
                        combs.append((mode, model, _cfg_env, s_wt, depth))
            scan_res = []
            checkpoint = time.time()
            time_for_scan = init_budget - (checkpoint - start) - (time_required_for_agent_setup) - (300)
            heuristic_fn = partial(heuristics.compute_score_stats, domain_path, instance_path, g_obs_keys, ga_keys, ac_dict_fn, n_episodes=5, time=time_for_scan)

            # JAX doesn't fork with fork context which is default for Linux. Start a spawn context explicitly.
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(mp_context=context) as executor:
                jobs = [executor.submit(heuristic_fn, copy.deepcopy(cfg), mode, model, _cfg_env, s_wt, depth) for (mode, model, _cfg_env, s_wt, depth) in combs]

                for job in as_completed(jobs):
                    result = job.result()
                    scan_res.append(result)

            scan_res = sorted(scan_res, key=lambda x: (x[0]))
            
            better_mode, better_weight, better_depth = scan_res[-1][1], scan_res[-1][2], scan_res[-1][3] 
            checkpoint = time.time()
            print(f"[Time left: {init_budget - (checkpoint - start)}] Heuristic scan complete.")

            if better_mode != cfg["mode"] or better_weight != cfg["logic_kwargs"]["weight"] or better_depth != cfg["depth"]:
                cfg["mode"] = better_mode
                cfg["logic_kwargs"]["weight"] = better_weight
                cfg["depth"] = better_depth
                if cfg["mode"] == "sampling":
                    new_agent = ContinuousDisprod(cfg, rddl_model, cfg_env)
                    new_lrs_to_scan = agent.pre_warm(dummy_obs)
                else:
                    new_agent = ContinuousDisprod(cfg, reparam_rddl_model, reparam_cfg_env)
                    new_lrs_to_scan = agent.pre_warm(dummy_obs)
                agent = new_agent
                if cfg[cfg["mode"]]["overwrite_lrs"]:
                    lrs_to_scan = cfg[cfg["mode"]]["lrs_to_scan"]
                else:
                    lrs_to_scan = new_lrs_to_scan
                checkpoint = time.time()
                print(f"[Time left: {init_budget - (checkpoint - start)}] Found better config during scan. New agent initialized.")
        
        
        ##############################################################
    except:
        finish = time.time()
        print('Initialization timed out', finish - start, ' seconds)')
        # print('This domain will continue exclusively with default actions!')
        init_timed_out = True

    signal.signal(signal.SIGALRM, signal_handler)
    
    for episode in range(episodes):
        total_reward = 0
        state = myEnv.reset()
        # timed_out = False if init_timed_out==False else True
        timed_out = False
        elapsed = budget
        start = 0
        for step in range(myEnv.horizon):

            # action selection:
            if not timed_out:
                signal.setitimer(signal.ITIMER_REAL, elapsed)
                start = time.time()
                try:
                    #################################################################
                    # replace the following line of code with your agent call
                    obs_array = np.array([state[i] for i in g_obs_keys])
                    ac_array, k_idx, prev_ac_seq, agent_key, _ = agent.choose_action(obs_array, prev_ac_seq, agent_key, lrs_to_scan)
                    action = ac_dict_fn(ac_array, k_idx)

                    #################################################################
                    finish = time.time()
                except:
                    finish = time.time()
                    print('Timed out! (', finish-start, ' seconds)')
                    print('This episode will continue with default actions!')
                    action = defaultAgent.sample_action()
                    timed_out = True
                    elapsed = 0
                    if not timed_out:
                        elapsed = elapsed - (finish-start)
            else:
                action = defaultAgent.sample_action()

            next_state, reward, done, info = myEnv.step(action)
            total_reward += reward

            print()
            print(f'step       = {step}')
            print(f'state      = {state}')
            print(f'action     = {action}')
            print(f'next state = {next_state}')
            print(f'reward     = {reward}')

            state = next_state

            if done:
                break

        print(f'episode {episode+1} ended with reward {total_reward} after {budget-elapsed} seconds')

    myEnv.close()



    ########################################
    # CLEAN UP ANY RESOURCES YOU HAVE USED #


    ########################################


# Command line interface, DO NOT CHANGE
if __name__ == "__main__":
    args = sys.argv
    print(args)
    method_name = None
    episodes = 1
    if len(args) == 2:
        if args[0] == '-h':
            print('python GymExample.py <domain> <instance> <method name> <num episodes>')
    if len(args) < 3:
        env, inst = 'HVAC', '1'
    elif len(args) < 4:
        env, inst = args[1:3]
    elif len(args) < 5:
        env, inst, method_name = args[1:4]
    else:
        env, inst, method_name, episodes = args[1:5]
        try:
            episodes = int(episodes)
        except:
            raise ValueError("episode must be an integer value argument, received: " + episodes)
    main(env, inst, method_name, episodes)

