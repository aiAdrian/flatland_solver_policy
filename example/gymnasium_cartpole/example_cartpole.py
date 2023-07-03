from environment.gymnasium.gymnasium_environment import GymnasiumEnvironment
from example.gymnasium_cartpole.cartpole_analytical_policy import CartpoleAnalyticalPolicy
from rendering.gymnasium.gymnasium_renderer import GymnasiumRenderer
from solver.gymnasium.gymnasium_solver import GymnasiumSolver
from utils.training_evaluation_pipeline import *


def create_cartpole_analytical_policy() -> Policy:
    return CartpoleAnalyticalPolicy()


if __name__ == "__main__":
    env = GymnasiumEnvironment("CartPole-v1")

    execute_policy_comparison(env, GymnasiumSolver)

    solver_analytical = GymnasiumSolver(env, create_cartpole_analytical_policy())
    solver_analytical.perform_evaluation(max_episodes=100)

    solver_analytical_rendering = GymnasiumSolver(env, create_cartpole_analytical_policy(), GymnasiumRenderer(env))
    solver_analytical_rendering.perform_evaluation(max_episodes=1)
