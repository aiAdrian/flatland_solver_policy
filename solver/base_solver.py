import os
from collections import deque
from typing import Union, Dict

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from environment.environment import Environment
from policy.policy import Policy
from rendering.base_renderer import BaseRenderer


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

    def activate_renderer(self, renderer: BaseRenderer):
        self.renderer = renderer
        self.activate_rendering()

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
            action = policy.act(handle, state, eps)

        state_next, reward, terminal, info = env.step(action)

        tot_terminal = 0
        for handle in self.env.get_agent_handles():
            all_terminal = all_terminal & terminal[handle]
            tot_terminal += terminal[handle]
            tot_reward += reward[handle]

            policy.step(handle, state, action, reward[handle], state_next, terminal[handle])
        tot_terminal /= max(1.0, len(self.env.get_agent_handles()))

        policy.end_step(train=training_mode)
        return state_next, tot_reward, all_terminal, tot_terminal, info

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
        tot_terminal = 0
        tot_steps = 0
        while True and tot_steps < self.max_steps:
            if self.before_step_starts():
                return tot_reward

            state_next, reward, terminal, tot_terminal, info = self.run_step(env,
                                                                             policy,
                                                                             state,
                                                                             eps,
                                                                             info,
                                                                             training_mode)
            tot_reward += reward
            state = self.update_state(state_next)
            self.render(episode, tot_steps, terminal)

            if self.after_step_ends():
                return tot_reward

            if terminal:
                break

            tot_steps += 1

        return tot_reward, tot_terminal, tot_steps

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
        tot_reward, tot_terminate, tot_steps = self.run_internal_episode(episode,
                                                                         env,
                                                                         policy,
                                                                         state,
                                                                         eps,
                                                                         info,
                                                                         training_mode)
        policy.end_episode(train=training_mode)
        self.after_episode_ends()
        return tot_reward, tot_terminate, tot_steps

    def perform_evaluation(self,
                           max_episodes=2000,
                           eps=0.0,
                           write_summary=True,
                           checkpoint_interval=50):

        training_mode = False

        episode = 0
        scores_window = deque(maxlen=checkpoint_interval)
        terminate_window = deque(maxlen=checkpoint_interval)
        nbr_agents_window = deque(maxlen=checkpoint_interval)
        tot_steps_window = deque(maxlen=checkpoint_interval)

        writer = None
        if write_summary:
            writer = SummaryWriter(comment="_" + self.get_name() + "_evaluation_" + self.policy.get_name())

        while True:
            episode += 1

            tot_reward, tot_terminate, tot_steps = self.run_episode(episode, self.env, self.policy, eps, training_mode)

            scores_window.append(tot_reward)
            terminate_window.append(tot_terminate)
            nbr_agents_window.append(self.env.get_num_agents())
            tot_steps_window.append(tot_steps)

            b = int(np.round(50 * np.mean(terminate_window)))
            done_bar = ['#'] * b + ['_'] * (50 - b)

            print(
                '\rEpisode: {:5}\treward: {:9.3f} : {:9.3f}  \tdone: [{:^5.0f}/{:^5.0f}] : {:4.3f}  \t [{}]'.format(
                    episode,
                    tot_reward,
                    np.mean(scores_window),
                    tot_terminate * self.env.get_num_agents(), self.env.get_num_agents(),
                    np.mean(terminate_window),
                    ''.join(list(done_bar)),
                ),
                end='\n' if episode % checkpoint_interval == 0 else '')

            if writer is not None:
                writer.add_scalar(self.get_name() + "/evaluation_value_reward", tot_reward, episode)
                writer.add_scalar(self.get_name() + "/evaluation_smoothed_reward", np.mean(scores_window), episode)
                writer.add_scalar(self.get_name() + "/evaluation_value_done", tot_terminate, episode)
                writer.add_scalar(self.get_name() + "/evaluation_smoothed_done", np.mean(terminate_window), episode)
                writer.add_scalar(self.get_name() + "/evaluation_value_nbr_agents", self.env.get_num_agents(), episode)
                writer.add_scalar(self.get_name() + "/evaluation_smoothed_nbr_agents", np.mean(nbr_agents_window),
                                  episode)
                writer.add_scalar(self.get_name() + "/evaluation_value_nbr_steps", tot_steps, episode)
                writer.add_scalar(self.get_name() + "/evaluation_smoothed_nbr_steps", np.mean(tot_steps_window),
                                  episode)

                writer.flush()

            if episode >= max_episodes:
                break

        print('\ndone.')

    def perform_training(self,
                         max_episodes=2000,
                         eps=1.0,
                         eps_decay=0.995,
                         min_eps=0.001,
                         checkpoint_interval=100):

        training_mode = True

        episode = 0
        scores_window = deque(maxlen=checkpoint_interval)
        terminate_window = deque(maxlen=checkpoint_interval)
        nbr_agents_window = deque(maxlen=checkpoint_interval)
        tot_steps_window = deque(maxlen=checkpoint_interval)

        scores_window.extend([0] * checkpoint_interval)
        terminate_window.extend([0] * checkpoint_interval)
        nbr_agents_window.extend([0] * checkpoint_interval)
        tot_steps_window.extend([0] * checkpoint_interval)

        writer = SummaryWriter(comment="_" + self.get_name() + "_training_" + self.policy.get_name())

        while True:
            episode += 1

            tot_reward, tot_terminate, tot_steps = self.run_episode(episode, self.env, self.policy, eps, training_mode)
            eps = max(min_eps, eps * eps_decay)

            scores_window.append(tot_reward)
            terminate_window.append(tot_terminate)
            nbr_agents_window.append(self.env.get_num_agents())
            tot_steps_window.append(tot_steps)

            b = int(np.round(50 * np.mean(terminate_window)))
            done_bar = ['#'] * b + ['_'] * (50 - b)

            print(
                '\rEpisode: {:5}\treward: {:9.3f} : {:9.3f} \tdone: [{:^5.0f}/{:^5.0f}] : {:4.3f} \t [{}] \t eps: {:7.3f} '.format(
                    episode,
                    tot_reward,
                    np.mean(scores_window),
                    tot_terminate * self.env.get_num_agents(), self.env.get_num_agents(),
                    np.mean(terminate_window),
                    ''.join(list(done_bar)),
                    eps
                ),
                end='\n' if episode % checkpoint_interval == 0 else '')

            writer.add_scalar(self.get_name() + "/training_value_reward", tot_reward, episode)
            writer.add_scalar(self.get_name() + "/training_smoothed_reward", np.mean(scores_window), episode)
            writer.add_scalar(self.get_name() + "/training_value_done", tot_terminate, episode)
            writer.add_scalar(self.get_name() + "/training_smoothed_done", np.mean(terminate_window), episode)
            writer.add_scalar(self.get_name() + "/training_value_nbr_agents", self.env.get_num_agents(), episode)
            writer.add_scalar(self.get_name() + "/training_smoothed_nbr_agents", np.mean(nbr_agents_window), episode)
            writer.add_scalar(self.get_name() + "/training_value_nbr_steps", tot_steps, episode)
            writer.add_scalar(self.get_name() + "/training_smoothed_nbr_steps", np.mean(tot_steps_window), episode)
            writer.flush()

            if episode % checkpoint_interval == 0 or episode >= max_episodes:
                self.save_policy(filename="{}/{}_{}_{}".format(writer.get_logdir(),
                                                               self.get_name(), self.policy.get_name(),
                                                               episode))

            if episode >= max_episodes:
                break

        # --- end training --------------------------------------------------------------------------
        self.save_policy(None)
        print('\ndone.')

    def save_policy(self,
                    filename: Union[str, None] = None):
        """
        If the filename is None use default policy output destination and default name.
        """
        if filename is None:
            if not os.path.exists('training_output'):
                os.makedirs('training_output')
            filename = "training_output/{}_{}".format(self.get_name(), self.policy.get_name())
        if self.policy is not None:
            self.policy.save(filename)

    def load_policy(self,
                    filename: Union[str, None] = None):
        """
        If the filename is None use default policy output source location and default name.
        """
        if filename is None:
            filename = "training_output/{}_{}".format(self.get_name(), self.policy.get_name())
        if self.policy is not None:
            self.policy.load(filename)
