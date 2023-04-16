from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.rendertools import AgentRenderVariant, RenderTool

from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.dead_lock_avoidance_agent import \
    DeadlockAvoidanceObservation, DeadLockAvoidanceAgent
from policy.policy import Policy
from solver.base_renderer import BaseRenderer
from solver.flatland_solver import FlatlandSolver

import numpy as np


class SimpleRenderer(BaseRenderer):
    def __init__(self, rail_env: RailEnv):
        super(SimpleRenderer, self).__int__()
        self.env = rail_env
        self.renderer = self._create_renderer()
    def _create_renderer(self, show_debug=False,
                         agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                         screen_width_scale=40, screen_height_scale=25):
        renderer = RenderTool(self.env,
                                       agent_render_variant=agent_render_variant,
                                       show_debug=show_debug,
                                       screen_width=int(np.round(self.env.width * screen_width_scale)),
                                       screen_height=int(np.round(self.env.height * screen_height_scale)))
        renderer.reset()
        return renderer

    def reset(self):
        self.renderer.reset

    def render(self):
        self.renderer.render_env(
            show=True,
            frames=False,
            show_observations=False,
            show_predictions=False
        )


def create_flatland_env(max_rails_between_cities=2,
                        max_rails_in_city=4,
                        malfunction_rate=1 / 1000,
                        n_cities=5,
                        number_of_agents=10,
                        grid_width=30,
                        grid_height=40,
                        random_seed=0) -> RailEnv:
    return RailEnv(
        width=grid_width,
        height=grid_height,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            seed=random_seed,
            grid_mode=True,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rails_in_city
        ),
        malfunction_generator=ParamMalfunctionGen(
            MalfunctionParameters(
                malfunction_rate=malfunction_rate, min_duration=10, max_duration=50
            )
        ),
        random_seed=random_seed,
        number_of_agents=number_of_agents,
        obs_builder_object=DeadlockAvoidanceObservation()
    )

def create_environment():
    environment = create_flatland_env()
    observation_space = None
    action_space = environment.action_space
    return environment, observation_space, action_space


def create_deadlock_avoidance_policy(rail_env: RailEnv, action_space: int, show_debug_plot=False) -> Policy:
    return DeadLockAvoidanceAgent(rail_env, action_space, enable_eps=False, show_debug_plot=show_debug_plot)


if __name__ == "__main__":
    env, obs_space, act_space = create_environment()
    renderer = SimpleRenderer(env)

    solver = FlatlandSolver(env)
    solver.set_renderer(renderer)
    # solver.activate_rendering()
    solver.set_policy(create_deadlock_avoidance_policy(env, act_space, False))
    solver.do_training(max_episodes=1000)
