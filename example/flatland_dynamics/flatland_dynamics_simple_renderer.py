import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import AgentRenderVariant, RenderTool
from flatland_railway_extension.utils.FlatlandDynamicsRenderer import FlatlandDynamicsRenderer

from solver.base_renderer import BaseRenderer


class FlatlandDynamicsSimpleRenderer(BaseRenderer):
    def __init__(self, rail_env: RailEnv, render_each_episode=1, render_each_step=10):
        super(FlatlandDynamicsSimpleRenderer, self).__int__()
        self.env = rail_env
        self.renderer: FlatlandDynamicsRenderer = self._create_renderer()
        self.render_each_episode = render_each_episode
        self.render_each_step = render_each_step

    def _create_renderer(self, show_debug=False,
                         agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX):
        flatland_renderer = FlatlandDynamicsRenderer(
            env=self.env,
            show_debug=show_debug,
            agent_render_variant=agent_render_variant,
            show_agents=False,
            fix_aspect_ration=False)
        return flatland_renderer

    def reset(self):
        if self.renderer.is_closed():
            self.renderer: FlatlandDynamicsRenderer = self._create_renderer()
        self.renderer.env_renderer.reset()

    def render(self, episode, step, terminal):
        if not terminal and (episode - 1) % self.render_each_episode != 0:
            return
        if step % self.render_each_step == 0:
            self.renderer.render(
                show=True,
                show_observations=True,
                show_predictions=False,
                disable_background_rendering=False,
                show_rowcols=False
            )
