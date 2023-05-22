from example.ma_gym.integration import MultiAgentGymEnvironment, MultiAgentGymSolver, MultiAgentGymRenderer
from policy.learning_policy.dddqn_policy.dddqn_policy import DDDQN_Param, DDDQNPolicy
from policy.learning_policy.ppo_policy.ppo_agent import PPOPolicy

from policy.policy import Policy


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


env = MultiAgentGymEnvironment(env_to_load='ma_gym:TrafficJunction10-v0')
solver_ppo = MultiAgentGymSolver(env,
                                 create_ppo_policy(env.get_observation_space(), env.get_action_space()))
solver_ppo.perform_training(max_episodes=1000)
solver_ppo.perform_evaluation(max_episodes=100)

solver_dddqn = MultiAgentGymSolver(env,
                                   create_dddqn_policy(env.get_observation_space(), env.get_action_space()))
solver_dddqn.perform_training(max_episodes=1000)
solver_dddqn.perform_evaluation(max_episodes=100)

solver_dddqn.set_and_activate_renderer(MultiAgentGymRenderer(env))
solver_dddqn.perform_evaluation(max_episodes=100)
