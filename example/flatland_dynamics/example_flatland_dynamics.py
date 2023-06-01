from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from environment.environment import Environment
from environment.flatland_railway_extension.flatland_dynamics import FlatlandDynamicsEnvironment
from observation.flatland.flatten_tree_observation_for_rail_env.flatten_tree_observation_for_rail_env import \
    FlattenTreeObsForRailEnv
from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.deadlock_avoidance_policy import \
    DeadLockAvoidancePolicy
from policy.policy import Policy
from rendering.flatland_railway_extension.flatland_dynamics_simple_renderer import FlatlandDynamicsSimpleRenderer
from solver.flatland_railway_extension.flatland_dynamics_solver import FlatlandDynamicsSolver
from training_evaluation_pipeline import experimental_training_evaluation_pipeline


def create_deadlock_avoidance_policy(environment: Environment, action_space: int, show_debug_plot=False) -> Policy:
    return DeadLockAvoidancePolicy(environment.get_raw_env(), action_space, enable_eps=False,
                                   show_debug_plot=show_debug_plot)


if __name__ == "__main__":
    observation_builder = FlattenTreeObsForRailEnv(
        max_depth=3,
        predictor=ShortestPathPredictorForRailEnv(max_depth=50)
    )

    env = FlatlandDynamicsEnvironment(obs_builder_object=observation_builder,
                                      number_of_agents=10)
    print('{} : agents: {:3} actions: {:3} obs_space: {:4}'.format(env.get_name(),
                                                                   env.get_num_agents(),
                                                                   env.get_action_space(),
                                                                   env.get_observation_space()))

    experimental_training_evaluation_pipeline(env, FlatlandDynamicsSolver)

    solver_deadlock = FlatlandDynamicsSolver(env, create_deadlock_avoidance_policy(env, env.get_action_space(), False),
                                             FlatlandDynamicsSimpleRenderer(env, render_each_episode=1))
    solver_deadlock.perform_training(max_episodes=2)
