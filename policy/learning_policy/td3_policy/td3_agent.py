import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from policy.learning_policy.learning_policy import LearningPolicy
from policy.learning_policy.replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidsize=256):
        super(Actor, self).__init__()
        self.device = device
        self.state_size = state_size
        self.action_size = action_size

        self.model = nn.Sequential(
            nn.Linear(state_size, hidsize),
            nn.Tanh(),
            nn.Linear(hidsize, hidsize),
            nn.Tanh(),
            nn.Linear(hidsize, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.model(state)


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidsize=256):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Q1 architecture
        self.model_q1 = nn.Sequential(
            nn.Linear(state_size + action_size, hidsize),
            nn.LeakyReLU(),
            nn.Linear(hidsize, hidsize),
            nn.LeakyReLU(),
            nn.Linear(hidsize, 1),
            nn.LeakyReLU()
        )

        # Q2 architecture
        self.model_q2 = nn.Sequential(
            nn.Linear(state_size + action_size, hidsize),
            nn.LeakyReLU(),
            nn.Linear(hidsize, hidsize),
            nn.LeakyReLU(),
            nn.Linear(hidsize, 1),
            nn.LeakyReLU()
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.model_q1(sa)
        q2 = self.model_q2(sa)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.model_q1(sa)


# Twin Delayed Deep Deterministic Policy Gradients (TD3)
class TD3Policy(LearningPolicy):
    def __init__(
            self,
            state_size,
            action_size,
            buffer_size=32_000,
            batch_size=256,
            discount=0.95,
            learning_rate=0.5e-3,
            tau=0.02,
            policy_noise=0.5,
            noise_clip=1.0,
            policy_freq=2
    ):
        super(TD3Policy, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.device = torch.device("cpu")
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer_min_size = 0
        self.learning_rate = learning_rate

        self.actor = Actor(state_size, action_size).to(self.device)
        self.actor_target = Actor(state_size, action_size).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        self.critic = Critic(state_size, action_size).to(self.device)
        self.critic_target = Critic(state_size, action_size).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)
        self.loss = 0
        self.total_it = 0

        self.t_step = 0
        self.update_every = 5

    def get_name(self):
        return self.__class__.__name__

    def act(self, handle, state, eps=0.):
        state = torch.FloatTensor(np.array(state).reshape(1, -1)).to(self.device)

        # Epsilon-greedy action selection
        if np.random.random() >= eps:
            estimated = self.actor(state)
            return np.argmax(estimated.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.action_size))

    def step(self, handle, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            self._train_net()

    def _train_net(self):
        if len(self.memory) <= self.buffer_min_size or len(self.memory) <= self.batch_size:
            return

        self.total_it += 1

        # Sample replay buffer
        states, actions, rewards, states_next, dones, _ = self.memory.sample()
        actions = F.one_hot(torch.squeeze(actions), num_classes=self.action_size)
        states_next = torch.squeeze(states_next)
        rewards = torch.squeeze(rewards)
        dones = torch.squeeze(dones)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn(states_next.shape[0], states_next.shape[1]) * self.policy_noise).clamp(
                -self.noise_clip,
                self.noise_clip)
            noisy_actions = self.actor_target(states_next + noise)
            noisy_action_greedy = torch.argmax(noisy_actions, dim=1)
            next_actions = F.one_hot(noisy_action_greedy, num_classes=self.action_size)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(states_next, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = torch.reshape(rewards, (len(rewards), 1)) + \
                       torch.reshape(1.0 - dones, (len(dones), 1)) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
                      F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            self._soft_update(self.critic, self.critic_target, self.tau)
            self._soft_update(self.actor, self.actor_target, self.tau)

        self.loss = critic_loss.mean().detach().cpu().numpy()

    def _soft_update(self, local_model, target_model, tau):
        # Soft update model parameters.
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
