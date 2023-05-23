import sys
import signal
import time
# sys.path.append('/home/test/pyRDDLGym')
import numpy as np
from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Policies.Agents import NoOpAgent

# for JAX backend:
# from pyRDDLGym.Core.Jax.JaxRDDLSimulator import JaxRDDLSimulator

############################################################
# IMPORT THE AGENT AND OTHER DEPENDENCIES OF YOUR SOLUTION #
from utils import helpers
from functools import partial
from planners.disprod import ContinuousDisprod
import jax
from utils.common_utils import prepare_config, load_method
import os


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
        rddl_model = helpers.gen_model(EnvInfo.get_domain(), EnvInfo.get_instance(inst), reparam_rddl)

        checkpoint = time.time()
        print(f"[Time: {checkpoint-start}] Reparam RDDL set to {reparam_rddl}")

        reward_fn, cpfs, const_dict, s_keys, a_keys, ga_keys, ns_keys, levels, grounded_names, extra_params, bool_s_idx, bool_a_idx, bool_ga_idx, real_ga_idx = helpers.prepare_rddl_compilations(rddl_model)

        checkpoint = time.time()
        print(f"[Time: {checkpoint-start}] Parsed RDDL Model")

        init_subs = myEnv.sampler.subs

        # obs_keys = state_keys - noise_keys. g_obs_keys = grounded version of obs_keys
        obs_keys = [key for key in s_keys if key not in DISPROD_NOISE_VARS]
        g_obs_keys = [k for k_ in obs_keys for k in grounded_names[k_]]
        
        # Map state/action to the indices capturing the grounded states/action in the transition function input vector
        s_gs_idx = helpers.prepare_index_mapping(obs_keys, grounded_names, init_subs, noise_vars=True)
        a_ga_idx = helpers.prepare_index_mapping(a_keys, grounded_names, init_subs, noise_vars=False)

        checkpoint = time.time()
        print(f"[Time: {checkpoint-start}] Generated index mappings")

        partial_transition_fn = helpers.ns_and_reward(cpfs, obs_keys + DISPROD_NOISE_VARS, a_keys, ns_keys, const_dict, levels, extra_params, reward_fn, s_gs_idx, a_ga_idx)

        cfg_env = {}

        cfg_env['transition_fn'] = partial_transition_fn
        cfg_env['ga_keys'] = ga_keys
        cfg_env['const_dict'] = const_dict
        cfg_env['bool_s_idx'] = bool_s_idx
        cfg_env['bool_ga_idx'] = bool_ga_idx
        cfg_env['real_ga_idx'] = real_ga_idx
        cfg_env["action_space"] = myEnv.action_space
        cfg_env["n_concurrent_ac"] = myEnv.numConcurrentActions
        cfg_env["nA"] = len(myEnv.action_space)
        cfg_env["nS"] = len(myEnv.observation_space)

        # Setup the projection function.
        projection_fn = load_method(cfg["disprod"]["projection_fn"])
        
        # projection_fn is for a row of actions. vmap here works on the depth axis.
        # Loading the executable function in cfg_env rather than cfg to prevent clash with wandb sweep
        if args.env == "recsim":
            n_consumer = len(rddl_model.objects["consumer"])
            n_item = len(rddl_model.objects["item"]) 
            cfg_env["disprod"] = {"projection_fn":jax.vmap(projection_fn(len(bool_ga_idx), n_consumer, n_item), in_axes=(0), out_axes=(0))}
        else:
            cfg_env["disprod"] = {"projection_fn" : jax.vmap(projection_fn(len(bool_ga_idx)), in_axes=(0), out_axes=(0))}

        checkpoint=time.time()
        print(f"[Time: {checkpoint-start}] Loading the projection fn")

        agent = ContinuousDisprod(cfg, cfg_env)
        checkpoint = time.time()
        print(f"[Time: {checkpoint-start}] Agent initialization complete")

        agent_key = jax.random.PRNGKey(cfg["seed"])
        prev_ac_seq, agent_key = agent.reset(agent_key)
        checkpoint = time.time()
        print(f"[Time: {checkpoint-start}] Agent reset complete")
        ################################################################
    except:
        finish = time.time()
        print('Timed out! (', finish - start, ' seconds)')
        print('This domain will continue exclusively with default actions!')
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
                    # action = agent.sample_action()
                    obs_array = np.array([state[i] for i in g_obs_keys])
                    # replace the following line of code with your agent call
                    ac_array, k_idx, prev_ac_seq, prev_ac_seq = agent.choose_action(obs_array, prev_ac_seq, prev_ac_seq)
                    action = {ga_keys[idx]: float(ac_array[idx]) for idx in k_idx}


                    #################################################################
                    finish = time.time()
                    print(f"[Time: {finish-start}] Action generated {action}")
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

