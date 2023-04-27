from solver.base_renderer import BaseRenderer


class CartpoleRenderer(BaseRenderer):
    def __init__(self, environment):
        super(CartpoleRenderer, self).__int__()
        self.env = environment

    def render(self, episode, step, terminal):
        self.env.render()
