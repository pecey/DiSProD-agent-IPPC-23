import numpy as onp
from matplotlib import pyplot as plt
from matplotlib import animation
import importlib
import random
import jax.numpy as jnp
from jax import random as jax_random
from omegaconf import OmegaConf
import os
from datetime import date
import time
from pathlib import Path

# Argmax with random tie-breaks
# The 0th-restart is always set to the previous solution
def random_argmax(key, x, pref_idx=0):
    try:
        options = jnp.where(x == jnp.nanmax(x))[0]
        val = 0 if 0 in options else jax_random.choice(key, options)
    except:
        val = jax_random.choice(key, jnp.arange(len(x)))
        print(f"All restarts where NaNs. Randomly choosing {val}.")
    finally:
        return val


# Miscellaneous functions
# Dynamically load a function from a module
def load_method(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn

# Print to file or stdout
def print_(x, path=None, flush=True):
    if path is None:
        print(x)
    else:
        print(x, file=open(path, 'a'), flush=flush)


# https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
def save_frames_as_gif(frames, path='./', filename='gym_animation.gif', save_as_video=False):
    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(f"{path}/{filename}", writer='imagemagick', fps=60)
    if save_as_video:
        video_filename = filename.replace(".gif", ".mp4")
        anim.save(f"{path}/{video_filename}", extra_args=['-vcodec', 'libx264'])
    plt.close()
    print(f"Saving GIF in {path}/{filename}, frames : {len(frames)}")
    # plt.savefig(f"{path}{filename}_{idx}.png")


# Seed the libraries
def set_global_seeds(seed, env=None):
    onp.random.seed(seed)
    random.seed(seed)

    if env:
        env.seed(seed)
        env.action_space.seed(seed)


def load_config_if_exists(path, log_path):
    if os.path.exists(path):
        print_(f"Using config file : {path}", log_path)
        return OmegaConf.load(path)
    print_(f"Requested config file not found at : {path}. Skipping", log_path)
    return {}

def prepare_config(env_name, cfg_path=None, log_path=None):
    config_files = [f"{cfg_path}/default.yaml",
                    f"{cfg_path}/{env_name}.yaml"
                    ]
    config = {}
    for config_file in config_files:
        config = OmegaConf.merge(config, load_config_if_exists(config_file, log_path))
    return config

def update_config_with_args(cfg, args, base_path):
    # Non-boolean keys
    keys_to_update = ["seed", "log_file", "depth", "n_episodes", "alg", "inst"]
    
    for key in keys_to_update:
        if args.__contains__(key) and getattr(args,key) is not None:
            cfg[key] = getattr(args, key)
     
    # Update n_restarts in our planner
    if args.__contains__("n_restarts") and getattr(args,"n_restarts") is not None:
        cfg["disprod"]["n_restarts"] = getattr(args, "n_restarts")
        
    # Update n_samples in CEM/MPPI
    if args.__contains__("n_samples") and getattr(args,"n_samples") is not None:
        cfg[cfg["alg"]]["n_samples"] = getattr(args, "n_samples")
    
    # Update step-size in our planner        
    if args.__contains__("step_size") and getattr(args,"step_size") is not None:
        cfg["disprod"]["step_size"] = getattr(args, "step_size")
        
    if args.__contains__("step_size_var") and getattr(args,"step_size_var") is not None:
        cfg["disprod"]["step_size_var"] = getattr(args, "step_size_var")
        
    if args.__contains__("taylor_expansion_mode") and getattr(args,"taylor_expansion_mode") is not None:
        cfg["disprod"]["taylor_expansion_mode"] = getattr(args, "taylor_expansion_mode")

    # Boolean keys
    boolean_keys = ["render", "headless", "save_as_gif"]
    for key in boolean_keys:
        if args.__contains__(key) and getattr(args,key) is not None:
            cfg[key] = getattr(args, key).lower() == "true"

    # If run_name is set, the update in config. Else set default value to {running_mode}_{current_time}
    if args.__contains__("run_name") and getattr(args, "run_name"):
        cfg["run_name"] = args.run_name
    else:
        today = date.today()
        if cfg["mode"] is not None:
            cfg["run_name"] = f"{today.strftime('%y-%m-%d')}_{cfg['mode']}_{int(time.time())}"
        else:
            cfg["run_name"] = f"{today.strftime('%y-%m-%d')}_{int(time.time())}"

    return cfg

def setup_output_dirs(cfg, run_name, base_path):   
    base_dir = f"{base_path}/results/{cfg['env_name']}/planning/{run_name}"
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    cfg["results_dir"] = base_dir

    log_dir = f"{base_dir}/logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    cfg["log_dir"] = log_dir
    cfg["log_file"] = f"{log_dir}/debug.log"


def plan_one_step(agent, env, state):
    action, imagined_trajectory = agent.choose_action(state)
    return onp.array(action), imagined_trajectory

# Don't move this above. Can lead to circular import issues.
from planners.shooting_cem import ShootingCEM
from planners.discrete_disprod import DiscreteDisprod
from planners.continuous_disprod import ContinuousDisprod

def setup_planner(env, env_cfg, key , config_rddlsim):
    if env_cfg["alg"] in ["cem", "mppi"]:
        agent = ShootingCEM(env, env_cfg, key, config_rddlsim)    
    elif env_cfg["alg"] == "disprod":
        if env_cfg["discrete"]:
            agent = DiscreteDisprod(env, env_cfg, key, config_rddlsim)
        else:
            agent = ContinuousDisprod(env, env_cfg, key, config_rddlsim)
    return agent