import copy
import gym
from gym.spaces import Discrete, Dict, Box
import numpy as np
import pygame
import os

from pyRDDLGymHelper.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGymHelper.Core.ErrorHandling.RDDLException import RDDLTypeError, RDDLLogFolderError

from pyRDDLGymHelper.Core.Compiler.RDDLLiftedModel import RDDLLiftedModel
from pyRDDLGymHelper.Core.Debug.Logger import Logger, SimLogger
from pyRDDLGymHelper.Core.Env.RDDLConstraints import RDDLConstraints
from pyRDDLGymHelper.Core.Parser.parser import RDDLParser
from pyRDDLGymHelper.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGymHelper.Core.Simulator.RDDLSimulator import RDDLSimulator
from pyRDDLGym.Visualizer.TextViz import TextVisualizer


class RDDLEnv(gym.Env):
    
    def __init__(self, domain: str,
                 instance: str=None,
                 enforce_action_constraints: bool=False,
                 debug: bool=False,
                 log: bool=False,
                 simlogname: str=None,
                 backend: object=RDDLSimulator):
        '''Creates a new gym environment from the given RDDL domain + instance.
        
        :param domain: the RDDL domain
        :param instance: the RDDL instance
        :param enforce_action_constraints: whether to raise an exception if the
        action constraints are violated
        :param debug: whether to log compilation information to a log file
        :param log: whether to log simulation data to file
        :param backend: the subclass of RDDLSimulator to use as backend for
        simulation (currently supports numpy and Jax)
        '''
        super(RDDLEnv, self).__init__()
        self.domain_text = domain
        self.instance_text = instance
        self.enforce_action_constraints = enforce_action_constraints

        # time budget for applications limiting time on episodes.
        # hardcoded so cannot be changed externally.
        self.budget = 120

        # read and parse domain and instance
        reader = RDDLReader(domain, instance)
        domain = reader.rddltxt

        # parse RDDL file
        parser = RDDLParser(lexer=None, verbose=False)
        parser.build()
        rddl = parser.parse(domain)
        self.model = RDDLLiftedModel(rddl)
        
        # for logging
        ast = self.model._AST
        self.trial = 0
        log_fname = f'{ast.domain.name}_{ast.instance.name}'
        logger = Logger(f'{log_fname}_debug.log') if debug else None
        self.simlogger = None
        if log:
            curpath = os.path.abspath(__file__)
            for _ in range(3):
                curpath = os.path.split(curpath)[0]
            dir = os.path.join(curpath, 'Logs', simlogname, ast.domain.name)
            if not os.path.exists(dir):
                try:
                    os.makedirs(dir)
                except Exception as e:
                    if not isinstance(e, FileExistsError):
                        raise RDDLLogFolderError('Could not create log folder for domain ' + ast.domain.name + ' of method ' + simlogname + ' at path: ' + dir)

            simlog_fname = os.path.join(dir, ast.instance.name)
            self.simlogger = SimLogger(f'{simlog_fname}_log.csv')
        # self.simlogger = SimLogger(f'{log_fname}_log.csv') if log else None
        if self.simlogger:
            self.simlogger.clear(overwrite=False)
        
        # define the model sampler and bounds    
        self.sampler = backend(self.model, logger=logger)
        bounds = RDDLConstraints(self.sampler).bounds

        # set roll-out parameters
        self.horizon = self.model.horizon
        self.discount = self.model.discount
        self.max_allowed_actions = self.model.max_allowed_actions
            
        self.currentH = 0
        self.done = False

        # set default actions
        self.defaultAction = self.model.groundactions()

        # define the actions bounds
        self.actionsranges = self.model.groundactionsranges()
        action_space = Dict()
        for act in self.defaultAction:
            act_range = self.actionsranges[act]
            if act_range in self.model.enums:
                action_space[act] = Discrete(len(self.model.objects[act_range]))            
            elif act_range == 'real':
                action_space[act] = Box(low=bounds[act][0],
                                        high=bounds[act][1],
                                        dtype=np.float32)
            elif act_range == 'bool':
                action_space[act] = Discrete(2)
            elif act_range == 'int':
                high = bounds[act][1]
                if high == np.inf:
                    high = np.iinfo(np.int32).max
                low = bounds[act][0]
                if low == -np.inf:
                    low = np.iinfo(np.int32).min
                action_space[act] = Discrete(int(high - low + 1), start=int(low))
            else:
                raise RDDLTypeError(
                    f'Unknown action value type <{act_range}> in environment.')
        self.action_space = action_space

        # define the states bounds
        if self.sampler.isPOMDP:
            search_dict = self.model.groundobserv()
            ranges = self.model.groundobservranges()
        else:
            search_dict = self.model.groundstates()
            ranges = self.model.groundstatesranges()
            
        state_space = Dict()
        for state in search_dict:
            state_range = ranges[state]
            if state_range in self.model.enums:
                state_space[state] = Discrete(len(self.model.objects[state_range]))          
            elif state_range == 'real':
                state_space[state] = Box(low=bounds[state][0],
                                         high=bounds[state][1],
                                         dtype=np.float32)
            elif state_range == 'bool':
                state_space[state] = Discrete(2)
            elif state_range == 'int':
                high = bounds[state][1]
                if high == np.inf:
                    high = np.iinfo(np.int32).max
                low = bounds[state][0]
                if low == -np.inf:
                    low = np.iinfo(np.int32).min
                state_space[state] = Discrete(int(high - low + 1), start=int(low))
            else:
                raise RDDLTypeError(
                    f'Unknown state value type <{state_range}> in environment.')
        self.observation_space = state_space

        # set the visualizer
        # the next line should be changed for the default behaviour - TextVix
        self._visualizer = TextVisualizer(self.model)
        self._movie_generator = None
        self.state = None
        self.image = None
        self.window = None
        self.to_render = False
        self.image_size = None

    def set_visualizer(self, viz, movie_gen=None, movie_per_episode=False, **viz_kwargs):
        self._visualizer = viz(self.model, **viz_kwargs)
        self._movie_generator = movie_gen
        self._movie_per_episode = movie_per_episode
        self._movies = 0
        self.to_render = False

    def step(self, actions):
        if self.done:
            return self.state, 0.0, self.done, {}

        # make sure the action length is of currect size
        action_length = len(actions)
        if (action_length > self.max_allowed_actions):
            raise RDDLInvalidNumberOfArgumentsError(
                f'Invalid action, expected at most '
                f'{self.max_allowed_actions} entries, '
                f'but got {action_length}.')
        
        # set full action vector
        # values are clipped to be inside the feasible action space
        clipped_actions = copy.deepcopy(self.defaultAction)
        for act in actions:
            if str(self.action_space[act]) == 'Discrete(2)':
                if self.actionsranges[act] == 'bool':
                    clipped_actions[act] = bool(actions[act])
            else:
                clipped_actions[act] = actions[act]
                
        # check action constraints
        if self.enforce_action_constraints:
            self.sampler.check_action_preconditions(clipped_actions)
        
        # sample next state and reward
        obs, reward, self.done = self.sampler.step(clipped_actions)
        state = self.sampler.states
            
        # check if the state invariants are satisfied
        if not self.done:
            self.sampler.check_state_invariants()               

        # log to file
        if self.simlogger is not None:
            self.simlogger.log(
                obs, clipped_actions, reward, self.done, self.currentH)
        
        # update step horizon
        self.currentH += 1
        if self.currentH == self.horizon:
            self.done = True

        # for visualization purposes
        self.state = state

        return obs, reward, self.done, {}

    def reset(self):
        self.total_reward = 0
        self.currentH = 0
        obs, self.done = self.sampler.reset()
        self.state = self.sampler.states

        image = self._visualizer.render(self.state)
        if self._movie_generator is not None:
            if self._movie_per_episode:
                self._movie_generator.save_animation(
                    self._movie_generator.env_name + '_' + str(self._movies))
                self._movies += 1
            self._movie_generator.save_frame(image)            
        self.image_size = image.size

        # Logging
        if self.simlogger:
            self.trial += 1
            text = '######################################################\n'
            text += 'New Trial\n'
            text += '######################################################'
            self.simlogger.log_free(text)

        return obs

    def pilImageToSurface(self, pilImage):
        return pygame.image.fromstring(
            pilImage.tobytes(), pilImage.size, pilImage.mode).convert()

    def render(self, to_display=True):
        if self._visualizer is not None:
            image = self._visualizer.render(self.state)
            if to_display:
                if not self.to_render:
                    self.to_render = True
                    pygame.init()
                    self.window = pygame.display.set_mode(
                        (self.image_size[0], self.image_size[1]))
                self.window.fill(0)
                pygameSurface = self.pilImageToSurface(image)
                self.window.blit(pygameSurface, (0, 0))
                pygame.display.flip()
    
            if self._movie_generator is not None:
                self._movie_generator.save_frame(image)
    
        return image
    
    def close(self):
        if self.simlogger:
            self.simlogger.close()
                        
        if self.to_render:
            pygame.display.quit()
            pygame.quit()
    
            if self._movie_generator is not None:
                self._movie_generator.save_animation(
                    self._movie_generator.env_name + '_' + str(self._movies))
                self._movies += 1

    @property
    def numConcurrentActions(self):
        return self.max_allowed_actions
    
    @property
    def non_fluents(self):
        return self.model.groundnonfluents()

    @property
    def Budget(self):
        return self.budget
