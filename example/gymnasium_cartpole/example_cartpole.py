from environment.gymnasium.gymnasium_environment import GymnasiumEnvironment
from example.gymnasium_cartpole.cartpole_analytical_policy import CartpoleAnalyticalPolicy
from policy.policy import Policy
from rendering.gymnasium.gymnasium_renderer import GymnasiumRenderer
from solver.gymnasium.gymnasium_solver import GymnasiumSolver
from utils.training_evaluation_pipeline import create_ppo_policy, experimental_training_evaluation_pipeline


def create_cartpole_analytical_policy() -> Policy:
    return CartpoleAnalyticalPolicy()


if __name__ == "__main__":
    env = GymnasiumEnvironment("CartPole-v1")
    solver = GymnasiumSolver(env, create_ppo_policy(env.get_observation_space(), env.get_action_space()))

    experimental_training_evaluation_pipeline(env, GymnasiumSolver)

    solver = GymnasiumSolver(env, create_cartpole_analytical_policy())
    solver.perform_evaluation(max_episodes=100)

    solver = GymnasiumSolver(env, create_cartpole_analytical_policy(), GymnasiumRenderer(env))
    solver.perform_evaluation(max_episodes=1)
