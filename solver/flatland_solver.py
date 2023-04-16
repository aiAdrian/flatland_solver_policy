from collections import deque

import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import AgentRenderVariant, RenderTool
from torch.utils.tensorboard import SummaryWriter

from solver.base_renderer import BaseRenderer
from solver.base_solver import BaseSolver


class FlatlandSolver(BaseSolver):
    def __init__(self, env: RailEnv):
        super(FlatlandSolver, self).__init__(env)
        self.policy = None
        self.renderer: BaseRenderer = None

    def set_renderer(self, renderer: BaseRenderer):
        self.renderer = renderer
        self.activate_rendering()

    def render(self, episode, terminal):
        if self.rendering_enabled:
            self.renderer.render(episode, terminal)

    def reset(self):
        state, _ = self.env.reset()
        if self.policy is not None:
            self.policy.reset(self.env)
        if self.renderer is not None:
            self.renderer.reset()
        return self.transform_state(state)

    def transform_state(self, state):
        return state

    def update_state(self, state_next):
        return state_next

    def run_step(self, env, policy, state, eps, training_mode):
        tot_reward = 0
        all_terminal = True

        policy.start_step(train=training_mode)

        actions = {}
        for handle in self.env.get_agent_handles():
            policy.start_act(handle, train=training_mode)
            action = policy.act(handle,
                                state[handle],
                                eps)
            actions.update({handle: action})
            policy.end_act(handle, train=training_mode)

        raw_state_next, reward, terminal, info = env.step(actions)
        state_next = self.transform_state(raw_state_next)

        for handle in self.env.get_agent_handles():
            policy.step(handle,
                        state[handle],
                        actions[handle],
                        reward[handle],
                        state_next[handle],
                        terminal[handle])
            all_terminal = all_terminal & terminal[handle]
            tot_reward += reward[handle]

        policy.end_step(train=training_mode)
        return state_next, tot_reward, all_terminal
