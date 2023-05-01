from environment.gymnasium.cartpole import CartpoleEnvironment
from example.gymnasium_cartpole.cartpole_analytical_policy import CartpoleAnalyticalPolicy
from example.gymnasium_cartpole.cartpole_renderer import CartpoleRenderer
from example.gymnasium_cartpole.cartpole_solver import CartpoleSolver
from policy.learning_policy.dddqn_policy.dddqn_policy import DDDQNPolicy, DDDQN_Param
from policy.learning_policy.ppo_policy.ppo_agent import PPOPolicy
from policy.policy import Policy

def create_dddqn_policy(observation_space: int, action_space: int) -> Policy:
    param = DDDQN_Param(hidden_size=128,
                        buffer_size=5_000,
                        batch_size=512,
                        update_every=10,
                        learning_rate=0.5e-3,
                        tau=1.e-2,
                        gamma=0.95,
                        buffer_min_size=0,
                        use_gpu=False)

    return DDDQNPolicy(observation_space, action_space, param)


def create_ppo_policy(observation_space: int, action_space: int) -> Policy:
    return PPOPolicy(observation_space, action_space, True)


def create_cartpole_analytical_policy() -> Policy:
    return CartpoleAnalyticalPolicy()


if __name__ == "__main__":
    env = CartpoleEnvironment()
    solver = CartpoleSolver(env, create_ppo_policy(env.get_observation_space(), env.get_action_space()))
    solver.perform_training(max_episodes=1000)

    solver = CartpoleSolver(env, create_dddqn_policy(env.get_observation_space(), env.get_action_space()))
    solver.perform_training(max_episodes=1000)

    solver = CartpoleSolver(env, create_cartpole_analytical_policy(), CartpoleRenderer(env))
    solver.perform_training(max_episodes=2)
