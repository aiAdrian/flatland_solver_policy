from example.ma_gym.integration import MultiAgentGymEnvironment, MultiAgentGymSolver, MultiAgentGymRenderer
from policy.learning_policy.ppo_policy.ppo_agent import PPOPolicy

from policy.policy import Policy


def create_ppo_policy(observation_space: int, action_space: int) -> Policy:
    return PPOPolicy(observation_space, action_space, True)


env = MultiAgentGymEnvironment(env_to_load='ma_gym:PongDuel-v0')
solver = MultiAgentGymSolver(env,
                             create_ppo_policy(env.get_observation_space(), env.get_action_space()),
                             MultiAgentGymRenderer(env))
solver.deactivate_rendering()
solver.perform_training(max_episodes=1000)
solver.activate_rendering()
solver.perform_evaluation(max_episodes=100)


