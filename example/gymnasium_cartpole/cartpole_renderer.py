from solver.base_renderer import BaseRenderer
from solver.environment import Environment


class CartpoleRenderer(BaseRenderer):
    def __init__(self, environment: Environment):
        super(CartpoleRenderer, self).__int__()
        self.env = environment

    def render(self, episode, step, terminal):
        self.env.get_raw_env().render()
