import gym

from policy.learning_policy.dddqn_policy.dddqn_policy import DDDQNPolicy, DDDQN_Param
from policy.learning_policy.ppo_policy.ppo_agent import PPOPolicy
from policy.policy import Policy
from solver.base_renderer import BaseRenderer
from solver.base_solver import BaseSolver


class CartPoolAnalyticalPolicy(Policy):
    def get_name(self) -> str:
        return self.__class__.__name__

    def act(self, handle: int, state, eps=0.):
        t, w = state[2:4]
        if abs(t) < 0.03:
            return 0 if w < 0 else 1
        return 0 if t < 0 else 1

    def step(self, handle: int, state, action, reward, next_state, done):
        pass


class CartPoolRenderer(BaseRenderer):
    def __init__(self, environment):
        super(CartPoolRenderer, self).__int__()
        self.env = environment

    def render(self, episode, terminal):
        self.env.render()


def create_environment():
    environment = gym.make("CartPole-v1")
    observation_space = environment.observation_space.shape[0]
    action_space = environment.action_space.n
    return environment, observation_space, action_space


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


def create_cart_pool_analytical_policy() -> Policy:
    return CartPoolAnalyticalPolicy()


class CartPoolSolver(BaseSolver):
    def get_name(self) -> str:
        return self.__class__.__name__


if __name__ == "__main__":
    env, obs_space, act_space = create_environment()

    solver = CartPoolSolver(env)

    renderer = CartPoolRenderer(env)
    solver.set_renderer(renderer)


    solver.set_policy(create_cart_pool_analytical_policy())
    solver.do_training(max_episodes=1000)

    solver.deactivate_rendering()
    solver.set_policy(create_dddqn_policy(obs_space, act_space))
    solver.do_training(max_episodes=1000)

    solver.set_policy(create_ppo_policy(obs_space, act_space))
    solver.do_training(max_episodes=1000)
