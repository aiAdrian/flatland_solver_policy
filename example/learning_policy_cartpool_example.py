from collections import deque
from collections import namedtuple

import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from policy.learning_policy.dddqn_policy import DDDQNPolicy, DDDQN_Param
from policy.learning_policy.ppo_agent import PPOPolicy


class RLSolver:
    def __init__(self, env, observation_space, action_space):
        self.env = env
        self.observation_space = observation_space
        self.action_space = action_space

    def create_policy(self, observation_space, action_space):
        raise NotImplementedError


    def _reset(self, env, policy):
        state = env.reset()
        policy.reset(env)
        return state


    def _run_step(self, env, policy, state, eps, training_mode):
        handle = 0
        tot_reward = 0
        all_terminal = True
        policy.start_step(train=training_mode)

        policy.start_act(handle, train=training_mode)
        action = policy.act(handle, state, eps)
        policy.end_act(handle, train=training_mode)

        state_next, reward, terminal, info = env.step(action)

        all_terminal = all_terminal & terminal
        tot_reward += reward

        policy.step(handle, state, action, reward, state_next, terminal)

        policy.end_step(train=training_mode)
        return state_next, tot_reward, all_terminal


    def _run_internal_episode(self, env, policy, state, eps, training_mode):
        tot_reward = 0
        while True:
            state_next, reward, terminal = self._run_step(env, policy, state, eps, training_mode)
            tot_reward += reward
            state = np.copy(state_next)
            if terminal:
                break
        return tot_reward


    def run_episode(self, env, policy, eps, training_mode):
        state = self._reset(env, policy)

        policy.start_episode(train=training_mode)
        tot_reward = self._run_internal_episode(env, policy, state, eps, training_mode)
        policy.end_episode(train=training_mode)

        return tot_reward


    def do_training(self, max_episodes=2000):
        eps = 1.0
        eps_decay = 0.99
        min_eps = 0.01
        training_mode = True

        policy = self.create_policy(self.observation_space, self.action_space)

        episode = 0
        checkpoint_interval = 20
        scores_window = deque(maxlen=100)
        writer = SummaryWriter(comment="_" + policy.getName())

        while True:
            episode += 1

            tot_reward = self.run_episode(self.env, policy, eps, training_mode)
            eps = max(min_eps, eps * eps_decay)
            scores_window.append(tot_reward)

            print('\rEpisode: {:5}\treward: {:7.3f}\t avg: {:7.3f}'.format(episode,
                                                                           tot_reward,
                                                                           np.mean(scores_window)),
                      end='\n' if episode % checkpoint_interval == 0 else '')

            writer.add_scalar("CartPole/value", tot_reward, episode)
            writer.add_scalar("CartPole/smoothed_value", np.mean(scores_window), episode)
            writer.flush()

            if episode >= max_episodes:
                break


class DDDQNSolver(RLSolver):
    def create_policy(self, observation_space, action_space):
        dddqn_param = DDDQN_Param(hidden_size=128,
                                     buffer_size=1000,
                                     batch_size=64,
                                     update_every=10,
                                     learning_rate=1.e-3,
                                     tau=1.e-2,
                                     gamma=0.95,
                                     buffer_min_size=0,
                                     use_gpu=False)

        return DDDQNPolicy(observation_space, action_space, dddqn_param)

class PPOSolver(RLSolver):
    def create_policy(self, observation_space, action_space):
        return PPOPolicy(observation_space, action_space, True)

def create_environment():
    env = gym.make("CartPole-v1")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    return env, observation_space, action_space


if __name__ == "__main__":
    env, observation_space, action_space = create_environment()
    DDDQNSolver(env, observation_space, action_space).do_training(max_episodes=1000)
    PPOSolver(env, observation_space, action_space).do_training(max_episodes=1000)
