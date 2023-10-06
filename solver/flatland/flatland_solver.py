from typing import Union, Callable, List, Dict

from environment.environment import Environment
from policy.policy import Policy
from rendering.base_renderer import BaseRenderer
from solver.base_solver import BaseSolver

RewardList = List[float]
TerminalList = List[float]
InfoDict = Dict
FlatlandRewardShaper = Callable[[RewardList, TerminalList, InfoDict, Environment], List[float]]


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

            # choose action for agent (handle)
            action, updated = self.run_choose_action(eps, handle, info, policy, state)
            update_values[handle] = updated
            actions.update({handle: action})

            policy.end_act(handle, train=training_mode)

        # run simulation step (env)
        raw_state_next, reward, terminal, info = env.step(actions)

        # shape reward and transform observation (if required)
        reward = self.shape_reward(reward, terminal, info)
        state_next = self.transform_state(raw_state_next)

        # calculate total reward, terminal_all, ..
        tot_terminal = 0
        for handle in self.env.get_agent_handles():
            terminal_all &= terminal[handle]
            tot_reward += reward[handle]
            tot_terminal += int(terminal[handle])

        self.run_policy_step(actions, policy, reward, state, state_next, terminal, terminal_all, update_values)
        tot_terminal /= max(1.0, len(self.env.get_agent_handles()))

        policy.end_step(train=training_mode)
        return state_next, tot_reward, terminal['__all__'], tot_terminal, info

    def run_choose_action(self, eps, handle, info, policy, state):
        if info['action_required'][handle]:
            updated = True
            action = policy.act(handle,
                                state[handle],
                                eps)
        else:
            # An action is not required if the train hasn't joined the railway network,
            # if it already reached its target, or if the train is currently malfunctioning.
            updated = False
            action = 0
        return action, updated

    def run_policy_step(self, actions, policy, reward, state, state_next, terminal, terminal_all, update_values):
        for handle in self.env.get_agent_handles():
            if update_values[handle] or terminal_all:
                policy.step(handle,
                            state[handle],
                            actions[handle],
                            reward[handle],
                            state_next[handle],
                            terminal[handle])
