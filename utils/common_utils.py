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
    return OmegaConf.to_container(config)