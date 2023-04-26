from solver.base_renderer import BaseRenderer


class CartPoolRenderer(BaseRenderer):
    def __init__(self, environment):
        super(CartPoolRenderer, self).__int__()
        self.env = environment

    def render(self, episode, terminal):
        self.env.render()
