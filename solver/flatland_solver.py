from collections import deque

import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import AgentRenderVariant, RenderTool
from torch.utils.tensorboard import SummaryWriter

from solver.base_solver import BaseSolver


class FlatlandSolver(BaseSolver):
    def __init__(self, env: RailEnv):
        super(FlatlandSolver, self).__init__(env)
        self.policy = None
        self.renderer = None

    def _create_renderer(self, show_debug=False,
                         agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                         screen_width_scale=40, screen_height_scale=25):
        self.renderer = RenderTool(self.env,
                                       agent_render_variant=agent_render_variant,
                                       show_debug=show_debug,
                                       screen_width=int(np.round(self.env.width * screen_width_scale)),
                                       screen_height=int(np.round(self.env.height * screen_height_scale)))
        self.renderer.reset()

    def make_renderer(self, renderer_class=None):
        if renderer_class is None:
            pass
        self._create_renderer()

    def render(self, episode, terminal):
        if self.rendering_enabled:
            self.renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=False,
                    show_predictions=False
                )

    def reset(self, env, policy):
        state, _ = env.reset()
        policy.reset(env)
        if self.renderer is not None:
            self.renderer.reset()
        return state

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
                                self.transform_state(state)[handle],
                                eps)
            actions.update({handle: action})
            policy.end_act(handle, train=training_mode)

        state_next, reward, terminal, info = env.step(actions)
        for handle in self.env.get_agent_handles():
            policy.step(handle,
                        self.transform_state(state)[handle],
                        actions[handle],
                        reward[handle],
                        self.transform_state(state_next[handle]),
                        terminal[handle])
            all_terminal = all_terminal & terminal[handle]
            tot_reward += reward[handle]

        policy.end_step(train=training_mode)
        return state_next, tot_reward, all_terminal
