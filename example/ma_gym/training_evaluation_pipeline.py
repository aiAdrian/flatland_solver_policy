from environment.environment import Environment
from example.ma_gym.integration import MultiAgentGymSolver, MultiAgentGymRenderer
from policy.learning_policy.a2c_policy.a2c_agent import A2CPolicy
from policy.learning_policy.dddqn_policy.dddqn_policy import DDDQNPolicy, DDDQN_Param
from policy.learning_policy.ppo_policy.ppo_agent import PPOPolicy
from policy.policy import Policy
from policy.random_policy import RandomPolicy


def crate_random_policy(observation_space: int, action_space: int) -> Policy:
    return RandomPolicy(action_space)


def create_dddqn_policy(observation_space: int, action_space: int) -> Policy:
    param = DDDQN_Param(hidden_size=256,
                        buffer_size=8_192,
                        batch_size=512,
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


def execute(env: Environment, policy_creator,
            max_episodes=1000, max_evaluation_episodes=100,
            do_training=True, do_evaluation=True, do_render_evaluation=False):
    solver = MultiAgentGymSolver(env,
                                 policy_creator(env.get_observation_space(), env.get_action_space()))
    if do_training:
        solver.perform_training(max_episodes=max_episodes)
    if do_evaluation:
        if do_render_evaluation:
            renderer = MultiAgentGymRenderer(env)
            renderer.set_sleep_time(0.1)
            solver.activate_renderer(renderer)
        solver.perform_evaluation(max_episodes=max_evaluation_episodes)


def experimental_training_evaluation_pipeline(env: Environment):
    execute(env, crate_random_policy)
    execute(env, create_a2c_policy)
    execute(env, create_ppo_policy)
    execute(env, create_dddqn_policy)
