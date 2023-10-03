import sys
import time
import numpy as np
from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager

# for JAX backend:
# from pyRDDLGym.Core.Jax.JaxRDDLSimulator import JaxRDDLSimulator

############################################################
# IMPORT THE AGENT AND OTHER DEPENDENCIES OF YOUR SOLUTION #
from utils import helpers
from planners.disprod import ContinuousDisprod
import jax
from utils.common_utils import prepare_config
import os

DISPROD_NOISE_VARS = ["disprod_eps_norm", "disprod_eps_uni"]
############################################################


# MAIN INTERACTION LOOP #
def main(env, inst, method_name=None, episodes=1):
    print(f'preparing to launch instance {inst} of domain {env}...')

    # get the environment info
    EnvInfo = ExampleManager.GetEnvInfo(env)

    log = False if method_name is None else True
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(),
                            instance=EnvInfo.get_instance(inst),
                            enforce_action_constraints=False,
                            debug=True,
                            log=log,
                            simlogname=method_name)
                            # backend=JaxRDDLSimulator)
    
    ################################################################
    # Initialize your agent here:

    current_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = prepare_config("_".join(env.lower().split()), f"{current_dir}/config")

    print(f"Loading config from path {current_dir}/config")

    # Don't reparameterize the RDDL expressions if planner uses sampling
    domain_path = EnvInfo.get_domain()
    instance_path = EnvInfo.get_instance(inst)
    
    # Setup rddl_model for sampling mode, and reparam_rddl_model for NV and complete mode
    rddl_model = helpers.gen_model(domain_path, instance_path, False)
    reparam_rddl_model = helpers.gen_model(domain_path, instance_path, True)

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
    print(f"Basic agent initialized. Time taken: {time_required_for_agent_setup}")

    
    for episode in range(episodes):
        total_reward = 0
        state = myEnv.reset()
        start = 0
        for step in range(myEnv.horizon):
            obs_array = np.array([state[i] for i in g_obs_keys])

            ac_array, k_idx, prev_ac_seq, agent_key, _ = agent.choose_action(obs_array, prev_ac_seq, agent_key, lrs_to_scan)
            action = ac_dict_fn(ac_array, k_idx)

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

        print(f'episode {episode+1} ended with reward {total_reward}')

    myEnv.close()


# Command line interface, DO NOT CHANGE
if __name__ == "__main__":
    args = sys.argv
    if len(args) < 5:
        print('python run_gym.py <domain> <instance> <method name> <num episodes>')
    env, inst, method_name, episodes = args[1:5]
    main(env, inst, method_name, episodes)
   