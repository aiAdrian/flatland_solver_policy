from typing import Union

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland_railway_extension.FlatlandEnvironmentHelper import FlatlandEnvironmentHelper
from flatland_railway_extension.environments.FlatlandDynamics import FlatlandDynamics
from flatland_railway_extension.utils.FlatlandDynamicsRenderer import FlatlandDynamicsRenderer

from example.flatland_dynamics.flatland_dynamics_env import FlatlandDynamicsSolver
from example.flatland_dynamics.flatland_dynamics_simple_renderer import FlatlandDynamicsSimpleRenderer
from example.flatland_rail_env.flatland_simple_renderer import FlatlandSimpleRenderer
from observation.flatland.flatten_tree_observation_for_rail_env.flatten_tree_observation_for_rail_env import \
    FlattenTreeObsForRailEnv
from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.deadlock_avoidance_policy import \
    DeadLockAvoidancePolicy
from policy.learning_policy.dddqn_policy.dddqn_policy import DDDQN_Param, DDDQNPolicy
from policy.learning_policy.ppo_policy.ppo_agent import PPOPolicy
from policy.policy import Policy
from solver.flatland_solver import FlatlandSolver


def create_deadlock_avoidance_policy(rail_env: RailEnv, action_space: int, show_debug_plot=False) -> Policy:
    return DeadLockAvoidancePolicy(rail_env, action_space, enable_eps=False, show_debug_plot=show_debug_plot)


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


def create_environment(obs_builder_object: Union[ObservationBuilder, None] = None,
                       number_of_agents: int = 10):
    flatland_environment_helper = FlatlandEnvironmentHelper(rail_env=FlatlandDynamics,
                                                            number_of_agents=number_of_agents,
                                                            random_seed=2341,
                                                            obs_builder_object=obs_builder_object)

    action_space = flatland_environment_helper.get_rail_env().action_space[0]

    obs_states, _ = flatland_environment_helper.get_rail_env().reset()
    observation_space = len(obs_states[0])

    return flatland_environment_helper.get_rail_env(), observation_space, action_space


if __name__ == "__main__":
    observation_builder = FlattenTreeObsForRailEnv(
        max_depth=3,
        predictor=ShortestPathPredictorForRailEnv(max_depth=50)
    )

    env, obs_space, act_space = create_environment(obs_builder_object=observation_builder, number_of_agents=10)
    print('{} : agents: {:3} actions: {:3} obs_space: {:4}'.format(env.__class__.__name__,
                                                                   env.get_num_agents(), act_space, obs_space))

    solver = FlatlandDynamicsSolver(env)

    renderer = FlatlandDynamicsSimpleRenderer(env, render_each_episode=1)
    solver.set_renderer(renderer)

    solver.activate_rendering()
    solver.set_policy(create_deadlock_avoidance_policy(env, act_space, False))
    solver.do_training(max_episodes=10)

    solver.deactivate_rendering()
    solver.set_policy(create_ppo_policy(obs_space, act_space))
    solver.do_training(max_episodes=1000)

    solver.deactivate_rendering()
    solver.set_policy(create_dddqn_policy(obs_space, act_space))
    solver.do_training(max_episodes=1000)
