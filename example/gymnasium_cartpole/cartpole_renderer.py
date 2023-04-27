from solver.base_renderer import BaseRenderer


class cartpoleRenderer(BaseRenderer):
    def __init__(self, environment):
        super(cartpoleRenderer, self).__int__()
        self.env = environment

    def render(self, episode, step, terminal):
        self.env.render()
