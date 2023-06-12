from typing import Callable, Type, List

from environment.environment import Environment
from policy.learning_policy.a2c_policy.a2c_agent import A2CPolicy, A2C_Param
from policy.learning_policy.dddqn_policy.dddqn_policy import DDDQNPolicy, DDDQN_Param
from policy.learning_policy.learning_policy import LearningPolicy
from policy.learning_policy.ppo_policy.ppo_agent import PPOPolicy, PPO_Param
from policy.learning_policy.td3_policy.td3_agent import TD3Policy, T3D_Param
from policy.policy import Policy
from policy.random_policy import RandomPolicy
from rendering.base_renderer import BaseRenderer
from solver.base_solver import BaseSolver


def crate_random_policy(observation_space: int, action_space: int) -> Policy:
    return RandomPolicy(action_space)


def create_dddqn_policy(observation_space: int, action_space: int) -> LearningPolicy:
    param = DDDQN_Param(hidden_size=256,
                        buffer_size=32_000,
                        buffer_min_size=0,
                        batch_size=1024,
                        learning_rate=0.5e-3,
                        discount=0.95,
                        update_every=5,
                        tau=0.5e-2,
                        use_gpu=False)
    return DDDQNPolicy(observation_space, action_space, param)


def create_ppo_policy(observation_space: int, action_space: int) -> LearningPolicy:
    ppo_param = PPO_Param(hidden_size=256,
                          buffer_size=32_000,
                          buffer_min_size=0,
                          batch_size=1024,
                          learning_rate=0.5e-3,
                          discount=0.95,
                          use_replay_buffer=True,
                          use_gpu=False)
    return PPOPolicy(observation_space, action_space, ppo_param)


def create_a2c_policy(observation_space: int, action_space: int) -> LearningPolicy:
    a2c_param = A2C_Param(hidden_size=256,
                          learning_rate=0.5e-3,
                          discount=0.95,
                          clip_grad_norm=0.1,
                          use_gpu=False)

    return A2CPolicy(observation_space, action_space, a2c_param)


def create_td3_policy(observation_space: int, action_space: int) -> LearningPolicy:
    t3d_param = T3D_Param(hidden_size=256,
                          buffer_size=32_000,
                          buffer_min_size=0,
                          batch_size=1024,
                          learning_rate=0.5e-3,
                          discount=0.95,
                          tau=0.5e-2,
                          policy_freq=2,
                          policy_noise=0.2,
                          noise_clip=0.1,
                          use_gpu=False)

    return TD3Policy(observation_space, action_space, t3d_param)


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
                                     max_evaluation_episodes=25,
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
        solver.load_policy(None)
        solver.perform_evaluation(max_episodes=max_evaluation_episodes)


policy_creator_list: List[Callable[[int, int], Policy]] = [crate_random_policy,
                                                           create_td3_policy,
                                                           create_a2c_policy,
                                                           create_ppo_policy,
                                                           create_dddqn_policy]


def execute_policy_comparison(env: Environment,
                              solver_creator: Type[BaseSolver],
                              renderer: BaseRenderer = None,
                              pcl: List[Callable[[int, int], Policy]] = policy_creator_list):
    for policy_creator in pcl:
        execute_single_policy_experiment(env, solver_creator, policy_creator, renderer=renderer)


def execute_replay_policy_comparison(env: Environment,
                                     solver_creator: Type[BaseSolver],
                                     renderer: BaseRenderer = None,
                                     max_evaluation_episodes=10,
                                     pcl: List[Callable[[int, int], Policy]] = policy_creator_list):
    for policy_creator in pcl:
        execute_single_policy_experiment(env=env,
                                         solver_creator=solver_creator,
                                         policy_creator=policy_creator,
                                         max_evaluation_episodes=max_evaluation_episodes,
                                         do_training=False,
                                         do_evaluation=True,
                                         renderer=renderer)
