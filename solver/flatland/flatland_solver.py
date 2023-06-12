from typing import Union, Callable, List, Dict

from environment.environment import Environment
from policy.policy import Policy
from rendering.base_renderer import BaseRenderer
from solver.base_solver import BaseSolver

FlatlandRewardShaper = Callable[[List[float], List[bool], Dict, Environment], List[float]]


class FlatlandSolver(BaseSolver):
    def __init__(self,
                 env: Environment,
                 policy: Policy,
                 renderer: Union[BaseRenderer, None] = None):
        super(FlatlandSolver, self).__init__(env, policy, renderer)
        self._reward_shaper: Union[FlatlandRewardShaper, None] = None

    def get_name(self) -> str:
        return self.__class__.__name__

    def reset(self):
        state, info = self.env.reset()
        if self.policy is not None:
            self.policy.reset(self.env)
        if self.renderer is not None:
            self.renderer.reset()
        return self.transform_state(state), info

    def transform_state(self, state):
        return state

    def update_state(self, state_next):
        return state_next

    def set_reward_shaper(self, reward_shaper: FlatlandRewardShaper):
        self._reward_shaper = reward_shaper

    def shape_reward(self, reward, terminal, info):
        if self._reward_shaper is not None:
            return self._reward_shaper(reward, terminal, info, self.env)
        return reward

    def run_step(self, env, policy, state, eps, info, training_mode):
        tot_reward = 0

        policy.start_step(train=training_mode)

        actions = {}
        update_values = [False] * env.get_num_agents()
        terminal_all = True
        for handle in self.env.get_agent_handles():
            policy.start_act(handle, train=training_mode)

            if info['action_required'][handle]:
                update_values[handle] = True
                action = policy.act(handle,
                                    state[handle],
                                    eps)
            else:
                # An action is not required if the train hasn't joined the railway network,
                # if it already reached its target, or if is currently malfunctioning.
                update_values[handle] = False
                action = 0

            actions.update({handle: action})
            policy.end_act(handle, train=training_mode)

        raw_state_next, reward, terminal, info = env.step(actions)
        reward = self.shape_reward(reward, terminal, info)
        state_next = self.transform_state(raw_state_next)

        for handle in self.env.get_agent_handles():
            terminal_all &= terminal[handle]

        for handle in self.env.get_agent_handles():
            if update_values[handle] or terminal_all:
                policy.step(handle,
                            state[handle],
                            actions[handle],
                            reward[handle],
                            state_next[handle],
                            terminal[handle])
            tot_reward += reward[handle]

        policy.end_step(train=training_mode)
        return state_next, tot_reward, terminal['__all__'], info
