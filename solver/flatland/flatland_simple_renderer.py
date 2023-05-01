import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import AgentRenderVariant, RenderTool

from solver.base_renderer import BaseRenderer
from solver.environment import Environment


class FlatlandSimpleRenderer(BaseRenderer):
    def __init__(self, rail_env: Environment, render_each_episode=1):
        super(FlatlandSimpleRenderer, self).__int__()
        self.env = rail_env
        self.renderer = self._create_renderer()
        self.render_each_episode = render_each_episode

    def _create_renderer(self, show_debug=False,
                         agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                         screen_width_scale=40, screen_height_scale=25):
        render_tool = RenderTool(self.env.get_raw_env(),
                                 agent_render_variant=agent_render_variant,
                                 show_debug=show_debug,
                                 screen_width=int(np.round(self.env.get_raw_env().width * screen_width_scale)),
                                 screen_height=int(np.round(self.env.get_raw_env().height * screen_height_scale)))
        render_tool.reset()
        return render_tool

    def reset(self):
        self.renderer.reset()

    def render(self, episode, step, terminal):
        if not terminal and (episode - 1) % self.render_each_episode != 0:
            return
        self.renderer.render_env(
            show=True,
            frames=False,
            show_observations=True,
            show_predictions=False
        )
