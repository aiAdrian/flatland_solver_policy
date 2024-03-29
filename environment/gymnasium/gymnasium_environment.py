from typing import List

import gym

from environment.environment import Environment


class GymnasiumEnvironment(Environment):

    def __init__(self,
                 env_to_load,
                 silent=False):
        self.env_to_load = env_to_load
        environment = gym.make(self.env_to_load)
        observation_space = environment.observation_space.shape[0]
        action_space = environment.action_space.n

        super(GymnasiumEnvironment, self).__init__(environment, action_space, observation_space, silent)

    def close(self):
        self.get_raw_env().close()

    def get_name(self) -> str:
        return "Environment:{}".format(self.env_to_load)

    def reset(self):
        state = self.raw_env.reset()
        return state, {}

    def step(self, actions):
        state_next, reward, done, info = self.raw_env.step(actions)
        terminal = {}
        terminal.update({0: done})
        terminal['__all__'] = done
        return state_next, [reward], terminal, info

    def get_agent_handles(self) -> List[int]:
        return [0]

    def get_num_agents(self) -> int:
        return 1
