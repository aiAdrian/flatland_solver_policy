from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from environment.environment import Environment
from environment.flatland.rail_env import RailEnvironment
from observation.flatland.flatten_tree_observation_for_rail_env.flatten_tree_observation_for_rail_env import \
    FlattenTreeObsForRailEnv
from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.deadlock_avoidance_policy import \
    DeadLockAvoidancePolicy
from policy.policy import Policy
from rendering.flatland.flatland_simple_renderer import FlatlandSimpleRenderer
from solver.flatland.flatland_solver import FlatlandSolver
from utils.training_evaluation_pipeline import execute_policy_comparison


def create_deadlock_avoidance_policy(environment: Environment, action_space: int, show_debug_plot=False) -> Policy:
    return DeadLockAvoidancePolicy(environment.get_raw_env(), action_space, enable_eps=False,
                                   show_debug_plot=show_debug_plot)


if __name__ == "__main__":
    do_rendering = True

    observation_builder = FlattenTreeObsForRailEnv(
        max_depth=4,
        predictor=ShortestPathPredictorForRailEnv(max_depth=0)
    )

    env = RailEnvironment(obs_builder_object=observation_builder, number_of_agents=5)

    print('{} : agents: {:3} actions: {:3} obs_space: {:4}'.format(env.get_name(),
                                                                   env.get_num_agents(),
                                                                   env.get_action_space(),
                                                                   env.get_observation_space()))

    # execute_policy_comparison(env, FlatlandSolver)

    solver_deadlock = FlatlandSolver(env,
                                     create_deadlock_avoidance_policy(env, env.get_action_space()),
                                     FlatlandSimpleRenderer(env) if do_rendering else None)
    solver_deadlock.perform_training(max_episodes=2)
