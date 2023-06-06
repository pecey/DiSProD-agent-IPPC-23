import jax.numpy as jnp
import jax
import re
import numpy as np
from pyRDDLGymHelper.Core.Parser import parser as Rddlparser
from pyRDDLGymHelper.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGymHelper.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel

from functools import partial
from utils.common_utils import load_method

EPS_STR = {"normal": "disprod_eps_norm",
           "uniform": "disprod_eps_uni",
           "weibull": "disprod_eps_uni",
           "bernoulli": "disprod_eps_uni"}

# Order of the list is important.
DISPROD_NOISE_VARS = ["disprod_eps_norm", "disprod_eps_uni"]


def prepare_index_mapping(keys, grounded_names, init_subs, noise_vars=False):
    mapping = {}
    idx = 0
    for k in keys:
        shape = init_subs[k].shape
        n_obj = len(grounded_names[k])
        squeeze_fn = jnp.array if "___" in grounded_names[k][0] else jnp.squeeze
        mapping[k] = (idx, idx + n_obj, squeeze_fn, shape)
        idx = idx + n_obj
    for k in DISPROD_NOISE_VARS:
        mapping[k] = (idx, idx + 1, jnp.squeeze, ())
        idx = idx + 1
    return mapping   


def reparam_normal(groups):
    return reparam(groups, "normal")

def reparam_uniform(groups):
    return reparam(groups, "uniform")

def reparam_weibull(groups):
    return reparam(groups, "weibull")

def reparam_bernoulli(groups):
    return reparam(groups, "bernoulli")

def reparam(match, dist):
    eps_str = EPS_STR[dist]
    groups = match.groups()
    
    if len(groups) == 1:
        arg_1 = groups[0].strip()
    elif len(groups) == 2:
        arg_1, arg_2 = groups[0].strip(), groups[1].strip()
    else:
        raise Exception("Unknown parameterization encountered.")
    
    # N(m, s^2) = m + s * N(0, 1)
    if dist == "normal":
        reparam_str = f"({arg_1} + {eps_str} * {arg_2})"
        return reparam_str
    
    # U(a,b) = a + (b-a) * U(0,1)
    if dist == "uniform":
        reparam_str = f"({arg_1} + {eps_str} * ({arg_2} - {arg_1}))"
        return reparam_str
    
    # W(s, r) = r * (-ln(1 - U(0,1))) ** (1 / s)
    if dist == "weibull":
        reparam_str = f"({arg_2} * pow[-ln[(1 - {eps_str})], 1/{arg_1}])"
        return reparam_str
    
    if dist == "bernoulli":
        reparam_str = f"(({arg_1}) * {eps_str})"
        return reparam_str
    
    raise Exception(f"Reparam for {dist} not yet defined.")


def gen_model(domain_path, instance_path, reparam = False):
    rddltxt = RDDLReader(domain_path, instance_path).rddltxt
    rddlparser = Rddlparser.RDDLParser()
    rddlparser.build()
    if reparam:
        rddltxt = reparam_rddl(rddltxt)
    ast = rddlparser.parse(rddltxt)
    model = RDDLLiftedModel(ast)
    return model    

def prepare_rddl_compilations(model): 
    a_keys = list(model.actions.keys())
    s_keys = list(model.states.keys())

    bool_s_idx = [idx for idx,key in enumerate(s_keys) if model.statesranges[key] == "bool"]
    
    ground_a_keys = list(model.groundactions().keys())
    real_ga_idx = [idx for idx,key in enumerate(ground_a_keys) if model.groundactionsranges()[key] == "real"]
    bool_ga_idx = [idx for idx,key in enumerate(ground_a_keys) if model.groundactionsranges()[key] == "bool"]

    ns_keys = [f"{k}'" for k in s_keys if k not in DISPROD_NOISE_VARS]

    return s_keys, list(a_keys), ground_a_keys, ns_keys, bool_s_idx, bool_ga_idx, real_ga_idx

def reparam_rddl(rddltxt):   
    # For arg1, match everything except a comma. For arg2, match everything except ) followed by at max one ). For patterns like (?p)
    normal_pattern = re.compile('Normal\(([^,]+),\s*([^)]+[\)]?)\)')
    uniform_pattern = re.compile('Uniform\(([^,]+),\s*([^)]+[\)]?)\)')
    weibull_pattern = re.compile('Weibull\(([^,]+),\s*([^)]+[\)]?)\)')
    bernoulli_pattern = re.compile('Bernoulli\(([^)]+[\)]?)\)')
    
    for (dist, pattern, reparam_fn) in [("normal",normal_pattern, reparam_normal), 
                                      ("uniform",uniform_pattern, reparam_uniform), 
                                      ("weibull",weibull_pattern, reparam_weibull),
                                      ("bernoulli", bernoulli_pattern, reparam_bernoulli)]:
        if pattern.search(rddltxt) is not None:
            rddltxt = re.sub(pattern, reparam_fn, rddltxt)

            eps_str = EPS_STR[dist]


            # Insert state-fluent definition and CPF just once.
            eps_default = f"{eps_str} : {{ state-fluent, real, default = 0.0 }};"
            pvar_pattern = re.compile('pvariables[\s]*{')
            eps_match = pvar_pattern.search(rddltxt)
            rddltxt = f"{rddltxt[:eps_match.end()]} \n {eps_default} \n {rddltxt[eps_match.end():]}"
            
            eps_cpf_str = f"{eps_str}' = {eps_str};"
            cpf_pattern = re.compile('cpfs[\s]*{')
            eps_cpf_match = cpf_pattern.search(rddltxt)
            rddltxt = f"{rddltxt[:eps_cpf_match.end()]} \n {eps_cpf_str} \n {rddltxt[eps_cpf_match.end():]}"

    #print(f"RDDL after noise injection: {rddltxt}")
    return rddltxt


def prepare_cfg_env(env_name, myEnv, rddl_model, cfg):
    g_obs_keys = [key for key in rddl_model.groundstates().keys() if key not in DISPROD_NOISE_VARS]
    s_keys, a_keys, ga_keys, ns_keys, bool_s_idx, bool_ga_idx, real_ga_idx = prepare_rddl_compilations(rddl_model)
    init_subs = myEnv.sampler.subs

    # obs_keys = state_keys - noise_keys. g_obs_keys = grounded version of obs_keys
    obs_keys = [key for key in s_keys if key not in DISPROD_NOISE_VARS]
    
    # Map state/action to the indices capturing the grounded states/action in the transition function input vector
    s_gs_idx = prepare_index_mapping(obs_keys, rddl_model.grounded_names, init_subs, noise_vars=True)
    a_ga_idx = prepare_index_mapping(a_keys, rddl_model.grounded_names, init_subs, noise_vars=False)

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
    
    # projection_fn = load_method(cfg["projection_fn"])
    
    if "recsim" in env_name.lower():
        # n_consumer = len(rddl_model.objects["consumer"])
        n_item = len(rddl_model.objects["item"]) 
        # cfg_env["projection_fn"] = jax.vmap(partial(projection_fn, len(bool_ga_idx), n_consumer, n_item), in_axes=(0), out_axes=(0))
        ac_dict_fn = partial(prep_ac_dict_recsim, n_item)
    else:
        # cfg_env["projection_fn"] = jax.vmap(partial(projection_fn, len(bool_ga_idx)), in_axes=(0), out_axes=(0))
        ga_keys_output_mapping = {idx: ((key, lambda x: float(x)) if idx in real_ga_idx else (key, lambda x: int(x)))  for idx, key in enumerate(ga_keys)}    
        ac_dict_fn = partial(prep_ac_dict, ga_keys_output_mapping)
    

    return g_obs_keys,ga_keys,ac_dict_fn,cfg_env

def prep_ac_dict(ga_keys_output_mapping, ac_array, k_idx):
    action = {ga_keys_output_mapping[int(idx)][0]: ga_keys_output_mapping[int(idx)][1](ac_array[idx]) for idx in k_idx}
    return action

def prep_ac_dict_recsim(n_item, ac_array, k_idx):
    best_ac = np.argmax(ac_array)
    con = int(best_ac/n_item)
    item = best_ac - (n_item * con)
    action = {f"recommend___c{con+1}__i{item+1}": 1}
    return action
