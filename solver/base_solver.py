from collections import deque
from typing import Union, Dict

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from environment.environment import Environment
from policy.policy import Policy
from solver.base_renderer import BaseRenderer


class BaseSolver:
    def __init__(self,
                 env: Environment,
                 policy: Policy,
                 renderer: Union[BaseRenderer, None] = None):
        self.env = env
        self.policy: Policy = policy
        self.rendering_enabled = False
        self.renderer: Union[BaseRenderer, None] = renderer
        if renderer is not None:
            self.activate_rendering()

        self.max_steps = np.inf

    def get_name(self) -> str:
        raise NotImplementedError

    def activate_rendering(self):
        self.rendering_enabled = True

    def deactivate_rendering(self):
        self.rendering_enabled = False

    def set_max_steps(self,
                      steps: int):
        self.max_steps = steps

    def render(self,
               episode: int,
               step: int,
               terminal: bool):
        if self.rendering_enabled:
            self.renderer.render(episode, step, terminal)

    def reset(self):
        state, info = self.env.reset()
        self.policy.reset(self.env)
        return state, info

    def run_step(self,
                 env: Environment,
                 policy: Policy,
                 state,
                 eps: float,
                 info: Dict,
                 training_mode: bool):

        tot_reward = 0
        all_terminal = True
        policy.start_step(train=training_mode)

        for handle in self.env.get_agent_handles():
            policy.start_act(handle, train=training_mode)
            action = policy.act(handle, state, eps)
            policy.end_act(handle, train=training_mode)

            state_next, reward, terminal, info = env.step(action)

            all_terminal = all_terminal & terminal
            tot_reward += reward

            policy.step(handle, state, action, reward, state_next, terminal)

        policy.end_step(train=training_mode)
        return state_next, tot_reward, all_terminal, info

    def update_state(self,
                     state_next):
        return np.copy(state_next)

    def before_step_starts(self):
        return False

    def after_step_ends(self):
        return False

    def run_internal_episode(self,
                             episode: int,
                             env: Environment, policy: Policy,
                             state,
                             eps: float,
                             info: Dict,
                             training_mode: bool):
        tot_reward = 0
        step = 0
        while True and step < self.max_steps:
            if self.before_step_starts():
                return tot_reward

            state_next, reward, terminal, info = self.run_step(env, policy, state, eps, info, training_mode)
            tot_reward += reward
            state = self.update_state(state_next)
            self.render(episode, step, terminal)

            if self.after_step_ends():
                return tot_reward

            if terminal:
                break

            step += 1
        return tot_reward

    def before_episode_starts(self):
        pass

    def after_episode_ends(self):
        pass

    def run_episode(self,
                    episode: int,
                    env: Environment,
                    policy: Policy,
                    eps: float,
                    training_mode: bool):
        state, info = self.reset()
        self.before_episode_starts()
        policy.start_episode(train=training_mode)
        tot_reward = self.run_internal_episode(episode, env, policy, state, eps, info, training_mode)
        policy.end_episode(train=training_mode)
        self.after_episode_ends()
        return tot_reward

    def perform_evaluation(self,
                           max_episodes=2000,
                           eps=0.0):

        training_mode = False

        episode = 0
        checkpoint_interval = 50
        scores_window = deque(maxlen=100)
        writer = SummaryWriter(comment=self.get_name() + "_eval_" + self.policy.get_name())

        while True:
            episode += 1

            tot_reward = self.run_episode(episode, self.env, self.policy, eps, training_mode)

            scores_window.append(tot_reward)

            print('\rEpisode: {:5}\treward: {:7.3f}\t avg: {:7.3f}'.format(episode,
                                                                           tot_reward,
                                                                           np.mean(scores_window)),
                  end='\n' if episode % checkpoint_interval == 0 else '')

            writer.add_scalar(self.get_name() + "/eval_value", tot_reward, episode)
            writer.add_scalar(self.get_name() + "/eval_smoothed_value", np.mean(scores_window), episode)
            writer.flush()

            if episode >= max_episodes:
                break

        print(' >> done.')

    def perform_training(self,
                         max_episodes=2000,
                         eps=1.0,
                         eps_decay=0.99,
                         min_eps=0.01):

        training_mode = True

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

            if episode % checkpoint_interval == 0:
                self.save_policy(filename=self.get_name() + "_" + self.policy.get_name() + "_" + episode)

        print(' >> done.')

    def save_policy(self,
                    filename: str):
        if self.policy is not None:
            self.policy.save(filename)

    def load_policy(self,
                    filename: str):
        if self.policy is not None:
            self.policy.load(filename)
