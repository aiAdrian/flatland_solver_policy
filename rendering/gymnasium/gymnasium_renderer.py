from environment.environment import Environment
from rendering.base_renderer import BaseRenderer


class GymnasiumRenderer(BaseRenderer):
    def __init__(self, environment: Environment):
        super(GymnasiumRenderer, self).__int__()
        self.env = environment

    def render(self, episode, step, terminal):
        self.env.get_raw_env().render()
