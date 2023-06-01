from environment.environment import Environment
from policy.learning_policy.a2c_policy.a2c_agent import A2CPolicy
from policy.learning_policy.dddqn_policy.dddqn_policy import DDDQNPolicy, DDDQN_Param
from policy.learning_policy.ppo_policy.ppo_agent import PPOPolicy
from policy.learning_policy.td3_policy.td3_agent import TD3Policy
from policy.policy import Policy
from policy.random_policy import RandomPolicy
from rendering.ma_gym.multi_agent_gym_renderer import MultiAgentGymRenderer


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


def execute(env: Environment, solver_creator, policy_creator,
            max_episodes=100, max_evaluation_episodes=100,
            do_training=True, do_rendering=False):
    print('=' * 61)
    solver = solver_creator(env,
                            policy_creator(env.get_observation_space(), env.get_action_space()))
    print('-' * 61)

    if do_rendering:
        renderer = MultiAgentGymRenderer(env)
        renderer.set_sleep_time(0.1)
        solver.activate_renderer(renderer)

    if do_training:
        print('- Training')
        solver.perform_training(max_episodes=max_episodes)

    print('- Evaluation')
    solver.perform_evaluation(max_episodes=max_evaluation_episodes)


def experimental_training_evaluation_pipeline(env: Environment, solver_creator):
    execute(env, solver_creator, crate_random_policy)
    execute(env, solver_creator, create_td3_policy)
    execute(env, solver_creator, create_a2c_policy)
    execute(env, solver_creator, create_ppo_policy)
    execute(env, solver_creator, create_dddqn_policy)
