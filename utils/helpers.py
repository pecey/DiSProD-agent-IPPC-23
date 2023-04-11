import jax.numpy as jnp
import re

from pyRDDLGym.Core.Parser import parser as Rddlparser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGym.Core.Jax import JaxRDDLCompiler, JaxRDDLBackpropPlanner

DISPROD_NOISE_VARS = ["disprod_eps_norm", "disprod_eps_uni"]

def ns_and_reward(cpfs, s_keys, a_keys, ns_keys, const_dict, levels, extra_params, reward_fn, s_gs_idx, a_ga_idx, state, action, rng_key):
    """
    s_keys, a_keys: not grounded 
    gs_keys, ga_keys: grounded
    grounded_names: map s_keys -> gs_keys, a_keys -> ga_keys
    state, action: grounded
    """

    state_dict = {k: state[s_gs_idx[k][0] : s_gs_idx[k][1]].squeeze() for k in s_keys}
    action_dict = {k: action[a_ga_idx[k][0] : a_ga_idx[k][1]].squeeze() for k in a_keys}
    
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
        mapping[k] = (idx, idx + n_obj)
        idx = idx + n_obj
    for k in DISPROD_NOISE_VARS:
        mapping[k] = (idx, idx + 1)
        idx = idx + 1
    return mapping    


def reparam(groups, dist):
    base_str = "disprod_eps"
    arg_1, arg_2 = groups[0].strip(), groups[1].strip()
    
    # N(m, s^2) = m + s * N(0, 1)
    if dist == "normal":
        eps_str = f"{base_str}_norm"
        reparam_str = f"({arg_1} + {eps_str} * {arg_2})"
        return eps_str, reparam_str

    if dist == "uniform":
        eps_str = f"{base_str}_uni"
        reparam_str = f"({arg_1} + {eps_str} * ({arg_2} - {arg_1}))"
        return eps_str, reparam_str
    
    # W(s, r) = r * (-ln(1 - eps_uni)) ** (1 / s)
    #TODO: Fix this. Using -ln(1) rather than -ln(1 - eps_uni)
    if dist == "weibull":
        eps_str = f"{base_str}_uni"
        reparam_str = f"({arg_2} * pow[-ln[(1 - {eps_str})], 1/{arg_1}])"
        return eps_str, reparam_str
    
    raise Exception(f"Reparam for {dist} not yet defined.")


def prepare_rddl_compilations(domain_path, instance_path): 
    # To read from pyRDDLGym
    # env_info = ExampleManager().GetEnvInfo(ENV)
    # domain = env_info.get_domain()
    # instance = env_info.get_instance(inst)        
    rddltxt = RDDLReader(domain_path, instance_path).rddltxt
    normal_pattern = re.compile('Normal\(([\s\w]+[-[\w]+]*),([\s\w]+[-[\w]+]*)\)')
    uniform_pattern = re.compile('Uniform\(([\s\w]+[-[\w]+]*),([\s\w]+[-[\w]+]*)\)')
    weibull_pattern = re.compile('Weibull\(([^,]+),\s*([^)]+[\)]?)\)')
    
    # TODO: How does this work for multiple  Normal()?
    for dist, pattern in [("normal",normal_pattern), ("uniform",uniform_pattern), ("weibull",weibull_pattern)]:
        match = pattern.search(rddltxt)
        if match is not None:
            groups = match.groups()
            eps_str, reparm_expr = reparam(groups, dist)
            rddltxt = rddltxt[:match.start()] + reparm_expr + rddltxt[match.end():]
            
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
    
    const_dict = {k:v for (k,v) in compiled.init_values.items() if k == k.upper()}

    return reward, cpfs, const_dict, s_keys, list(a_keys), ground_a_keys, ns_keys, levels, model.grounded_names, model_params