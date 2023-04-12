class Disprod:
    def __init__(self, cfg, key, config_rddlsim):
        self.a_keys = config_rddlsim.get('action_keys', [])
        self.const_dict = config_rddlsim.get('const_dict', {})
        self.var_dict = config_rddlsim.get('var_dict', {})
        self.cpfs = config_rddlsim.get('cpfs')
        self.dynamics_fn = config_rddlsim.get('transition_fn')
        self.reward_fn = config_rddlsim.get('reward_fn')

        self.nA = cfg["nA"]
        # +1 due to addition of noise variable as a pseduo-state variable.
        self.nS = cfg["nS"]
        
        self.depth = cfg.get("depth")
        self.device = cfg.get("device")
        self.alpha = cfg.get("alpha")
        self.save_actions = cfg.get("save_actions")
        self.debug = cfg.get("debug_planner")

        self.n_res = cfg["disprod"]["n_restarts"]
        self.max_grad_steps = cfg["disprod"]["max_grad_steps"]
        self.step_size = cfg["disprod"]["step_size"]
        self.step_size_var = cfg["disprod"]["step_size_var"]
        self.convergance_threshold = cfg["disprod"]["convergance_threshold"]
        self.run_mode = cfg['disprod']['run_mode']
        
        self.key = key

        # Noise parameters        
        self.norm_noise_mu = 0
        self.norm_noise_var = 1
        
        self.uni_noise_mu = 0
        self.uni_noise_var = 1/12
            
        self.saved_restart_action = None
        self.last_chosen_action = None
        self.promising_restart = None



    def reset(self):
        self.saved_restart_action = None
        self.last_chosen_action = None
        self.promising_restart = None
        
    def dynamics_fn_wrapper(self, state , action, key):
        # The order is important here. obs_dict should be after self.var_dict
        return self.dynamics_fn(state , action, key)
    
    def rewards_fn_wrapper(self, state, action, ns, key):
         # The order is important here. obs_dict should be after self.var_dict
        return self.reward_fn(state , action, ns, key)[0]