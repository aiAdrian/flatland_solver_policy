from typing import Optional, List

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.rendertools import AgentRenderVariant, RenderTool

from example.observation_utils import normalize_observation
from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.deadlock_avoidance_agent import \
    DeadLockAvoidanceAgent
from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.deadlock_avoidance_observation import \
    DeadlockAvoidanceObservation
from policy.learning_policy.dddqn_policy.dddqn_policy import DDDQN_Param, DDDQNPolicy
from policy.learning_policy.ppo_policy.ppo_agent import PPOPolicy
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

    def render(self, episode, terminal):
        if not terminal and episode % 50 != 0:
            return
        self.renderer.render_env(
            show=True,
            frames=False,
            show_observations=True,
            show_predictions=False
        )


class FlattenTreeObsForRailEnv(TreeObsForRailEnv):

    def _normalize_observation_tree_obs(self, agent_handle, raw_states, observation_radius=2):
        return normalize_observation(raw_states[agent_handle],
                                     self.max_depth,
                                     observation_radius=observation_radius)

    def _transform_state(self, states):
        ret_obs = None
        # Update replay buffer and train agent
        for agent_handle in self.env.get_agent_handles():

            # Preprocess the new observations
            if states[agent_handle]:
                normalized_obs = self._normalize_observation_tree_obs(agent_handle, states)
                if ret_obs is None:
                    ret_obs = np.zeros((self.env.get_num_agents(), len(normalized_obs)))

                ret_obs[agent_handle] = normalized_obs

        return ret_obs

    def get_many(self, handles: Optional[List[int]] = None):
        return self._transform_state(super(FlattenTreeObsForRailEnv, self).get_many(handles))


def create_flatland_env(obs_builder: ObservationBuilder,
                        max_rails_between_cities=2,
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
        obs_builder_object=obs_builder
    )


def create_environment(obs_builder: ObservationBuilder, number_of_agents: int):
    environment = create_flatland_env(obs_builder, number_of_agents=number_of_agents)
    action_space = environment.action_space[0]

    obs_states, _ = environment.reset()
    observation_space = len(obs_states[0])

    return environment, observation_space, action_space


def create_deadlock_avoidance_policy(rail_env: RailEnv, action_space: int, show_debug_plot=False) -> Policy:
    return DeadLockAvoidanceAgent(rail_env, action_space, enable_eps=False, show_debug_plot=show_debug_plot)


def create_dddqn_policy(observation_space: int, action_space: int) -> Policy:
    param = DDDQN_Param(hidden_size=128,
                        buffer_size=10_000,
                        batch_size=1024,
                        update_every=10,
                        learning_rate=0.5e-3,
                        tau=1.e-3,
                        gamma=0.95,
                        buffer_min_size=0,
                        use_gpu=False)

    return DDDQNPolicy(observation_space, action_space, param)


def create_ppo_policy(observation_space: int, action_space: int) -> Policy:
    return PPOPolicy(observation_space, action_space, True)


if __name__ == "__main__":
    observation_builder = FlattenTreeObsForRailEnv(
        max_depth=3,
        predictor=ShortestPathPredictorForRailEnv(max_depth=50)
    )

    env, obs_space, act_space = create_environment(observation_builder, number_of_agents=3)
    print('{} : agents: {:3} actions: {:3} obs_space: {:4}'.format(env, env.get_num_agents(), act_space, obs_space))


    solver = FlatlandSolver(env)

    renderer = SimpleRenderer(env)
    solver.set_renderer(renderer)
    solver.deactivate_rendering()

    solver.set_policy(create_ppo_policy(obs_space, act_space))
    solver.do_training(max_episodes=100)

    solver.set_policy(create_dddqn_policy(obs_space, act_space))
    solver.do_training(max_episodes=100)

    solver.activate_rendering()
    solver.set_policy(create_deadlock_avoidance_policy(env, act_space, False))
    solver.do_training(max_episodes=100)




