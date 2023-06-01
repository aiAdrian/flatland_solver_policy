import time

from environment.environment import Environment
from rendering.base_renderer import BaseRenderer


class MultiAgentGymRenderer(BaseRenderer):
    def __init__(self, environment: Environment):
        super(MultiAgentGymRenderer, self).__int__()
        self.env = environment
        self.sleep_time = 0.0

    def set_sleep_time(self, st):
        self.sleep_time = st

    def render(self, episode, step, terminal):
        self.env.get_raw_env().render()
        if self.sleep_time > 0:
            time.sleep(self.sleep_time)
