from typing import List, Union

import gym
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland_railway_extension.FlatlandEnvironmentHelper import FlatlandEnvironmentHelper
from flatland_railway_extension.environments.FlatlandDynamics import FlatlandDynamics


class Environment:

    def __init__(self, raw_env, action_space, observation_space):
        self.raw_env = raw_env
        self._action_space: int = action_space
        self._observation_space: int = observation_space

    def get_name(self):
        raise NotImplementedError  # return self.__class__.__name__

    def reset(self):
        # returns state : nd.array(n agents x state_size),
        #         info : {}
        # return self.raw_env.reset()
        raise NotImplementedError  #

    def step(self, actions):
        # returns state_next : nd.array(n agents x state_size),
        #         reward : array[float],
        #         terminal : array[bool],
        #         info : {}
        # state_next, reward, terminal, info = self.raw_env.step(actions)
        # return # state_next, reward, terminal, info
        raise NotImplementedError

    def get_observation_space(self) -> int:
        return self._observation_space

    def get_action_space(self) -> int:
        return self._action_space

    def get_raw_env(self):
        return self.raw_env

    def get_agent_handles(self) -> List[int]:
        return self.raw_env.get_agent_handles()

    def get_num_agents(self) -> int:
        return self.raw_env.get_num_agents()
