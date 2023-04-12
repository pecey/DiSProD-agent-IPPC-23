import sys
import signal
import time
import numpy as np


from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Policies.Agents import NoOpAgent


############################################################
# IMPORT THE AGENT AND OTHER DEPENDENCIES OF YOUR SOLUTION #
from utils import helpers
from functools import partial
from planners.continuous_disprod import ContinuousDisprod
import jax
from utils.common_utils import prepare_config
from planners.shooting_cem import ShootingCEM


############################################################

DISPROD_NOISE_VARS = ["disprod_eps_norm", "disprod_eps_uni"]

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
                            debug=False,
                            log=log,
                            simlogname=method_name)
    budget = myEnv.Budget

    # default noop agent, do not change
    defaultAgent = NoOpAgent(action_space=myEnv.action_space,
                        num_actions=myEnv.numConcurrentActions)

    ################################################################
    # Initialize your agent here:
    # remove the noop agent:

    reward_fn, cpfs, const_dict, s_keys, a_keys, ga_keys, ns_keys, levels, grounded_names, extra_params = helpers.prepare_rddl_compilations(EnvInfo.get_domain(), EnvInfo.get_instance(inst))

    obs_keys = [key for key in s_keys if key not in DISPROD_NOISE_VARS]
    g_obs_keys = [k for k_ in obs_keys for k in grounded_names[k_]]

    s_gs_idx = helpers.prepare_index_mapping(obs_keys, grounded_names, noise_vars=True)
    a_ga_idx = helpers.prepare_index_mapping(a_keys, grounded_names, noise_vars=False)

    ns_and_reward_fn = partial(helpers.ns_and_reward, cpfs, obs_keys + DISPROD_NOISE_VARS, a_keys, ns_keys, const_dict, levels, extra_params, reward_fn, s_gs_idx, a_ga_idx)

    config_rddlsim = {}
    config_rddlsim['transition_fn'] = ns_and_reward_fn
    config_rddlsim['cpfs'] = cpfs
    config_rddlsim['action_keys'] = ga_keys
    config_rddlsim['pyrddlgym'] = True
    config_rddlsim['const_dict'] = const_dict

    key = jax.random.PRNGKey(42)

    cfg = prepare_config("_".join(env.lower().split()), "config")
    
    cfg["action_space"] = myEnv.action_space
    cfg["nA"] = len(myEnv.action_space)
    cfg["nS"] = len(myEnv.observation_space)

    agent = ShootingCEM(cfg, key , config_rddlsim)

    # agent = NoOpAgent(action_space=myEnv.action_space,
    #                     num_actions=myEnv.numConcurrentActions)



    ################################################################


    signal.signal(signal.SIGALRM, signal_handler)

    for episode in range(episodes):
        total_reward = 0
        state = myEnv.reset()
        timed_out = False
        elapsed = budget + 10000
        finish = start = 0
        for step in range(myEnv.horizon):

            # action selection:
            if not timed_out:
                signal.setitimer(signal.ITIMER_REAL, elapsed)
                try:
                    start = time.time()
                    #################################################################
                    obs_array = np.array([state[i] for i in g_obs_keys])
                    # replace the following line of code with your agent call
                    action = agent.choose_action(obs_array)
                    action = {k : v.item() for k , v in zip(ga_keys , action)}




                    #################################################################
                    finish = time.time()
                except:
                    print('Timed out!')
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

