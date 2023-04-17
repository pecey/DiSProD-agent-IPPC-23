import jax.numpy as jnp
import re

from pyRDDLGym.Core.Parser import parser as Rddlparser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Jax import JaxRDDLCompiler, JaxRDDLBackpropPlanner

DISPROD_NOISE_VARS = ["disprod_eps_norm", "disprod_eps_uni"]

EPS_STR = {"normal": "disprod_eps_norm",
           "uniform": "disprod_eps_uni",
           "weibull": "disprod_eps_uni",
           "bernoulli": "disprod_eps_uni"}

def ns_and_reward(cpfs, s_keys, a_keys, ns_keys, const_dict, levels, extra_params, reward_fn, s_gs_idx, a_ga_idx, state, action, rng_key):
    """
    s_keys, a_keys: not grounded 
    gs_keys, ga_keys: grounded
    grounded_names: map s_keys -> gs_keys, a_keys -> ga_keys
    state, action: grounded
    """

    state_dict = {k: s_gs_idx[k][2](state[s_gs_idx[k][0] : s_gs_idx[k][1]]) for k in s_keys}
    action_dict = {k: a_ga_idx[k][2](action[a_ga_idx[k][0] : a_ga_idx[k][1]]) for k in a_keys}
    
    subs = {**state_dict, **action_dict, **const_dict}

    for key in levels:
        expr = cpfs[key]
        subs[key], rng_key, _ = expr(subs, extra_params, rng_key)
        
    reward, _, _ = reward_fn(subs, extra_params, rng_key)

    return jnp.hstack([subs[k] for k in ns_keys]), reward

def prepare_index_mapping(keys, grounded_names, noise_vars=False):
    mapping = {}
    idx = 0
    for k in keys:
        n_obj = len(grounded_names[k])
        squeeze_fn = jnp.array if "___" in grounded_names[k][0] else jnp.squeeze
        mapping[k] = (idx, idx + n_obj, squeeze_fn)
        idx = idx + n_obj
    for k in DISPROD_NOISE_VARS:
        mapping[k] = (idx, idx + 1, jnp.squeeze)
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


def prepare_rddl_compilations(domain_path, instance_path): 
    rddltxt = RDDLReader(domain_path, instance_path).rddltxt

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
                
    rddlparser = Rddlparser.RDDLParser()
    rddlparser.build()
    ast = rddlparser.parse(rddltxt)
    
    model = RDDLLiftedModel(ast)

    a_keys = model.actions.keys()
    s_keys = model.states.keys()
    
    ground_a_keys = model.groundactions().keys()

    ns_keys = [f"{k}'" for k in s_keys if k not in DISPROD_NOISE_VARS]

    compiled = JaxRDDLBackpropPlanner.JaxRDDLCompilerWithGrad(rddl=model)
    compiled.compile()

    reward, cpfs = compiled.reward, compiled.cpfs
    model_params = compiled.model_params
    
    levels = [_v for v in compiled.levels.values() for _v in v]  
    
    const_dict = {k:compiled.init_values[k] for k in compiled.rddl.nonfluents.keys()}

    return reward, cpfs, const_dict, s_keys, list(a_keys), ground_a_keys, ns_keys, levels, model.grounded_names, model_params