import gym

from solver.flatland_solver import FlatlandSolver
from policy.learning_policy.dddqn_policy.dddqn_policy import DDDQNPolicy, DDDQN_Param
from policy.learning_policy.ppo_policy.ppo_agent import PPOPolicy
from policy.policy import Policy


def create_environment():
    environment = gym.make("CartPole-v1")
    observation_space = environment.observation_space.shape[0]
    action_space = environment.action_space.n
    return environment, observation_space, action_space


def create_dddqn_policy(observation_space: int, action_space: int) -> Policy:
    param = DDDQN_Param(hidden_size=128,
                        buffer_size=10_000,
                        batch_size=512,
                        update_every=10,
                        learning_rate=0.5e-2,
                        tau=1.e-3,
                        gamma=0.95,
                        buffer_min_size=0,
                        use_gpu=False)

    return DDDQNPolicy(observation_space, action_space, param)


def create_ppo_policy(observation_space: int, action_space: int) -> Policy:
    return PPOPolicy(observation_space, action_space, True)


if __name__ == "__main__":
    env, obs_space, act_space = create_environment()

    solver = FlatlandSolver(env)

    solver.set_policy(create_dddqn_policy(obs_space, act_space))
    solver.do_training(max_episodes=1000)

    solver.set_policy(create_ppo_policy(obs_space, act_space))
    solver.do_training(max_episodes=1000)
