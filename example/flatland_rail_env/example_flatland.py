from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from environment.flatland.rail_env import RailEnvironment
from observation.flatland.flatten_tree_observation_for_rail_env.flatten_tree_observation_for_rail_env import \
    FlattenTreeObsForRailEnv
from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.deadlock_avoidance_policy import \
    DeadLockAvoidancePolicy
from policy.learning_policy.dddqn_policy.dddqn_policy import DDDQN_Param, DDDQNPolicy
from policy.learning_policy.ppo_policy.ppo_agent import PPOPolicy
from policy.policy import Policy
from environment.environment import Environment
from solver.flatland.flatland_simple_renderer import FlatlandSimpleRenderer
from solver.flatland.flatland_solver import FlatlandSolver


def create_deadlock_avoidance_policy(environment: Environment, action_space: int, show_debug_plot=False) -> Policy:
    return DeadLockAvoidancePolicy(environment.get_raw_env(), action_space, enable_eps=False,
                                   show_debug_plot=show_debug_plot)


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

    env = RailEnvironment(obs_builder_object=observation_builder, number_of_agents=10)

    print('{} : agents: {:3} actions: {:3} obs_space: {:4}'.format(env.get_name(),
                                                                   env.get_num_agents(),
                                                                   env.get_action_space(),
                                                                   env.get_observation_space()))

    solver_dddqn = FlatlandSolver(env, create_dddqn_policy(env.get_observation_space(), env.get_action_space()))
    solver_dddqn.perform_training(max_episodes=2)

    solver_ppo = FlatlandSolver(env, create_ppo_policy(env.get_observation_space(), env.get_action_space()))
    solver_ppo.perform_training(max_episodes=2)

    solver_deadlock = FlatlandSolver(env, create_deadlock_avoidance_policy(env, env.get_action_space()),
                            FlatlandSimpleRenderer(env))
    solver_deadlock.perform_training(max_episodes=2)