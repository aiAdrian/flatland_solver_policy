from collections import deque
from typing import Union

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from policy.policy import Policy
from solver.base_renderer import BaseRenderer


class BaseSolver:
    def __init__(self, env):
        self.env = env
        self.policy: Union[Policy, None] = None
        self.rendering_enabled = False
        self.renderer: Union[BaseRenderer, None] = None

    def get_name(self) -> str:
        return self.__class__.__name__

    def set_renderer(self, renderer: BaseRenderer):
        self.renderer = renderer

    def set_policy(self, policy: Policy):
        self.policy = policy

    def activate_rendering(self):
        self.rendering_enabled = True

    def deactivate_rendering(self):
        self.rendering_enabled = False

    def render(self, episode: int, terminal: bool):
        if self.rendering_enabled:
            self.renderer.render(episode, terminal)

    def reset(self):
        state = self.env.reset()
        self.policy.reset(self.env)
        return state, {}

    def run_step(self, env, policy, state, eps, info, training_mode):

        tot_reward = 0
        all_terminal = True
        policy.start_step(train=training_mode)

        for handle in [0]:
            policy.start_act(handle, train=training_mode)
            action = policy.act(handle, state, eps)
            policy.end_act(handle, train=training_mode)

            state_next, reward, terminal, info = env.step(action)

            all_terminal = all_terminal & terminal
            tot_reward += reward

            policy.step(handle, state, action, reward, state_next, terminal)

        policy.end_step(train=training_mode)
        return state_next, tot_reward, all_terminal, info

    def update_state(self, state_next):
        return np.copy(state_next)

    def run_internal_episode(self, episode, env, policy, state, eps, info, training_mode):
        tot_reward = 0
        while True:
            state_next, reward, terminal, info = self.run_step(env, policy, state, eps, info, training_mode)
            tot_reward += reward
            state = self.update_state(state_next)
            self.render(episode, terminal)
            if terminal:
                break
        return tot_reward

    def run_episode(self, episode, env, policy, eps, training_mode):
        state, info = self.reset()

        policy.start_episode(train=training_mode)
        tot_reward = self.run_internal_episode(episode, env, policy, state, eps, info, training_mode)
        policy.end_episode(train=training_mode)

        return tot_reward

    def do_training(self, max_episodes=2000):
        eps = 1.0
        eps_decay = 0.99
        min_eps = 0.01
        training_mode = True

        if self.policy is None:
            print('No policy set: please use set_policy(policy)')
            return

        episode = 0
        checkpoint_interval = 50
        scores_window = deque(maxlen=100)
        writer = SummaryWriter(comment=self.get_name() + "_" + self.policy.get_name())

        while True:
            episode += 1

            tot_reward = self.run_episode(episode, self.env, self.policy, eps, training_mode)
            eps = max(min_eps, eps * eps_decay)
            scores_window.append(tot_reward)

            print('\rEpisode: {:5}\treward: {:7.3f}\t avg: {:7.3f}'.format(episode,
                                                                           tot_reward,
                                                                           np.mean(scores_window)),
                  end='\n' if episode % checkpoint_interval == 0 else '')

            writer.add_scalar(self.get_name() + "/value", tot_reward, episode)
            writer.add_scalar(self.get_name() + "/smoothed_value", np.mean(scores_window), episode)
            writer.flush()

            if episode >= max_episodes:
                break

    def save_policy(self, filename: str):
        if self.policy is not None:
            self.policy.save(filename)

    def load_policy(self, filename: str):
        if self.policy is not None:
            self.policy.load(filename)
