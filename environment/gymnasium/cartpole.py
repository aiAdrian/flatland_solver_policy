from typing import List

import gym

from environment.environment import Environment


class CartpoleEnvironment(Environment):

    def __init__(self):
        print('ok')
        environment = gym.make("CartPole-v1")
        observation_space = environment.observation_space.shape[0]
        action_space = environment.action_space.n

        super(CartpoleEnvironment, self).__init__(environment, action_space, observation_space)

    def get_name(self) -> str:
        return "Environment:Gymnasium:CartPole-v1"

    def reset(self):
        state = self.raw_env.reset()
        return state, {}

    def step(self, actions):
        state_next, reward, terminal, info = self.raw_env.step(actions)
        return state_next, reward, terminal, info


    def get_agent_handles(self) -> List[int]:
        return [0]

    def get_num_agents(self) -> int:
        return 1
