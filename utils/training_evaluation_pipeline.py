from typing import Callable, Type

from environment.environment import Environment
from policy.learning_policy.a2c_policy.a2c_agent import A2CPolicy
from policy.learning_policy.dddqn_policy.dddqn_policy import DDDQNPolicy, DDDQN_Param
from policy.learning_policy.ppo_policy.ppo_agent import PPOPolicy
from policy.learning_policy.td3_policy.td3_agent import TD3Policy
from policy.policy import Policy
from policy.random_policy import RandomPolicy
from rendering.base_renderer import BaseRenderer
from solver.base_solver import BaseSolver


def crate_random_policy(observation_space: int, action_space: int) -> Policy:
    return RandomPolicy(action_space)


def create_dddqn_policy(observation_space: int, action_space: int) -> Policy:
    param = DDDQN_Param(hidden_size=256,
                        buffer_size=32_000,
                        batch_size=1024,
                        update_every=5,
                        learning_rate=0.5e-3,
                        tau=0.5e-2,
                        gamma=0.95,
                        buffer_min_size=0,
                        use_gpu=False)

    return DDDQNPolicy(observation_space, action_space, param)


def create_ppo_policy(observation_space: int, action_space: int) -> Policy:
    return PPOPolicy(observation_space, action_space, True)


def create_a2c_policy(observation_space: int, action_space: int) -> Policy:
    return A2CPolicy(observation_space, action_space)


def create_td3_policy(observation_space: int, action_space: int) -> Policy:
    return TD3Policy(observation_space, action_space)


def create_solver(env: Environment,
                  solver_creator: Type[BaseSolver],
                  policy_creator: Callable[[int, int], Policy],
                  renderer: BaseRenderer) -> BaseSolver:
    print('-' * 61)
    return solver_creator(env,
                          policy_creator(env.get_observation_space(), env.get_action_space()),
                          renderer)


def execute_single_policy_experiment(env: Environment,
                                     solver_creator: Type[BaseSolver],
                                     policy_creator: Callable[[int, int], Policy],
                                     max_episodes=1000,
                                     max_evaluation_episodes=1000,
                                     do_training=True,
                                     do_evaluation=True,
                                     renderer: BaseRenderer = None):
    """
    example usage:
        experimental_training_evaluation_pipeline(env, FlatlandSolver)
    """
    solver = create_solver(env, solver_creator, policy_creator, renderer)

    if do_training:
        print('-- Training')
        solver.perform_training(max_episodes=max_episodes)

    if do_evaluation:
        print('-- Evaluation')
        solver.perform_evaluation(max_episodes=max_evaluation_episodes)


def execute_policy_comparison(env: Environment,
                              solver_creator: Type[BaseSolver],
                              renderer: BaseRenderer = None):
    execute_single_policy_experiment(env, solver_creator, crate_random_policy, renderer=renderer)
    execute_single_policy_experiment(env, solver_creator, create_td3_policy, renderer=renderer)
    execute_single_policy_experiment(env, solver_creator, create_a2c_policy, renderer=renderer)
    execute_single_policy_experiment(env, solver_creator, create_ppo_policy, renderer=renderer)
    execute_single_policy_experiment(env, solver_creator, create_dddqn_policy, renderer=renderer)
