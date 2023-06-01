from typing import List

import gym
import numpy as np

from environment.environment import Environment


class MultiAgentGymEnvironment(Environment):

    def __init__(self, env_to_load):
        self.env_to_load = env_to_load
        environment = gym.make(self.env_to_load)
        obs_n = environment.reset()
        observation_space = len(obs_n[0])
        action_space = environment.action_space[0].n
        super(MultiAgentGymEnvironment, self).__init__(environment, action_space, observation_space)

    def get_name(self) -> str:
        return "Environment:{}".format(self.env_to_load)

    def reset(self):
        state = self.raw_env.reset()
        info = {'action_required': np.ones(self.get_num_agents())}
        return state, info

    def step(self, actions):
        transformed_actions = []
        for i in range(self.get_num_agents()):
            transformed_actions.append(actions.get(i))
        state_next, reward, dones, info = self.raw_env.step(transformed_actions)
        info['action_required'] = np.ones(self.get_num_agents())
        terminal = {}
        for i in range(self.get_num_agents()):
            terminal.update({i: dones[i]})
        terminal['__all__'] = np.sum(dones) == self.get_num_agents()
        return state_next, reward, terminal, info

    def get_agent_handles(self) -> List[int]:
        return list(range(self.get_num_agents()))

    def get_num_agents(self) -> int:
        return self.raw_env.n_agents
