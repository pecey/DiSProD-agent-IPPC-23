import jax
import jax.numpy as jnp
import re

from pyRDDLGymHelper.Core.Parser import parser as Rddlparser
from pyRDDLGymHelper.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGymHelper.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGymHelper.Core.Jax import JaxRDDLCompiler, JaxRDDLBackpropPlanner

EPS_STR = {"normal": "disprod_eps_norm",
           "uniform": "disprod_eps_uni",
           "weibull": "disprod_eps_uni",
           "bernoulli": "disprod_eps_uni"}

# Order of the list is important.
DISPROD_NOISE_VARS = ["disprod_eps_norm", "disprod_eps_uni"]

def ns_and_reward(cpfs, s_keys, a_keys, ns_keys, const_dict, levels, extra_params, reward_fn, s_gs_idx, a_ga_idx):
    def _ns_and_reward(state, action, rng_key):
        """
        s_keys, a_keys: not grounded 
        gs_keys, ga_keys: grounded
        grounded_names: map s_keys -> gs_keys, a_keys -> ga_keys
        state, action: grounded
        """

        # s_gs_idx and s_ga_idx have 4 values for each key.
        # idx 0 and idx 1 are the min and max indexes for the key
        # idx 3 is the desired shape.
        state_dict = {k: jnp.array(state[s_gs_idx[k][0] : s_gs_idx[k][1]]).reshape(s_gs_idx[k][3]) for k in s_keys}
        action_dict = {k: jnp.array(action[a_ga_idx[k][0] : a_ga_idx[k][1]]).reshape(a_ga_idx[k][3]) for k in a_keys}

        # subs should be not grounded.
        subs = {**state_dict, **action_dict, **const_dict}

        for level in levels:
            expr = cpfs[level]
            subs[level], _, _ = expr(subs, extra_params, rng_key)
            
        reward, _, _ = reward_fn(subs, extra_params, rng_key)

        # flatten is required in cases like RecSim where state variables are 2D
        return jnp.hstack([subs[k].flatten() for k in ns_keys]), reward
    return _ns_and_reward


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
    # To read from pyRDDLGym
    # env_info = ExampleManager().GetEnvInfo(ENV)
    # domain = env_info.get_domain()
    # instance = env_info.get_instance(inst)

    a_keys = list(model.actions.keys())
    s_keys = list(model.states.keys())

    bool_s_idx = [idx for idx,key in enumerate(s_keys) if model.statesranges[key] == "bool"]
    bool_a_idx = [idx for idx,key in enumerate(a_keys) if model.actionsranges[key] == "bool"]
    
    ground_a_keys = list(model.groundactions().keys())
    real_ga_idx = [idx for idx,key in enumerate(ground_a_keys) if model.groundactionsranges()[key] == "real"]
    bool_ga_idx = [idx for idx,key in enumerate(ground_a_keys) if model.groundactionsranges()[key] == "bool"]

    # ns_mapping = model.next_state
    # ns_keys = [ns_mapping[k] for k in s_keys if k not in ["disprod_epsilon"]]
    ns_keys = [f"{k}'" for k in s_keys if k not in DISPROD_NOISE_VARS]

    compiled = JaxRDDLBackpropPlanner.JaxRDDLCompilerWithGrad(rddl=model)
        
    # compiled_expr_tree = compiled._compile_cpfs_into_exp_tree()
    # for k,v in compiled_expr_tree.items():
    #     print(f"Key: {k}, Expression: {v}")
    
    compiled.compile()
        
    # JaxRDDLCompiler turns this on causing a lot of logs on screen. 
    jax.config.update('jax_log_compiles', False)


    reward, cpfs = compiled.reward, compiled.cpfs
    model_params = compiled.model_params
    
    # This decides the order of processing
    levels = [_v for v in compiled.levels.values() for _v in v]  
    
    # compiled.init_values is a dict of values of constants and variables. Split it into two dicts
    const_dict = {k:compiled.init_values[k] for k in compiled.rddl.nonfluents.keys()}

    return reward, cpfs, const_dict, s_keys, list(a_keys), ground_a_keys, ns_keys, levels, model.grounded_names, model_params, bool_s_idx, bool_a_idx, bool_ga_idx, real_ga_idx

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