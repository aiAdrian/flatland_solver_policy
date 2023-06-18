import copy
import os
from collections import namedtuple
from typing import Union

import torch
import torch.nn as nn

from policy.learning_policy.learning_policy import LearningPolicy
from policy.learning_policy.ppo_policy.ppo_agent import EpisodeBuffers

# Advantage Actor-Critic (A2C)
# https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html


class A2CActor(nn.Module):
    def __init__(self, state_size, action_size, device, hidsize1=512, hidsize2=256):
        super(A2CActor, self).__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(state_size, hidsize1),
            nn.Tanh(),
            nn.Linear(hidsize1, hidsize2),
            nn.Tanh(),
            nn.Linear(hidsize2, action_size),
            nn.Softmax(dim=-1)
        ).to(self.device)

    def forward(self, x):
        return self.model(x)

    def get_actor_dist(self, state):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs=probs)
        return dist, probs

    def save(self, filename):
        # print("Saving model from checkpoint:", filename)
        torch.save(self.model.state_dict(), filename + ".a2c_actor")

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            try:
                obj.load_state_dict(torch.load(filename, map_location=self.device))
            except:
                print(" >> failed!")
        return obj

    def load(self, filename):
        print("load model from file", filename)
        self.model = self._load(self.model, filename + ".a2c_actor")


# Critic module
class A2CCritic(nn.Module):
    def __init__(self, state_size, device, hidsize1=512, hidsize2=256):
        super(A2CCritic, self).__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(state_size, hidsize1),
            nn.Tanh(),
            nn.Linear(hidsize1, hidsize2),
            nn.Tanh(),
            nn.Linear(hidsize2, 1)
        ).to(self.device)

    def forward(self, x):
        return self.model(x)

    def save(self, filename):
        # print("Saving model from checkpoint:", filename)
        torch.save(self.model.state_dict(), filename + ".a2c_critic")

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            try:
                obj.load_state_dict(torch.load(filename, map_location=self.device))
            except:
                print(" >> failed!")
        return obj

    def load(self, filename):
        print("load model from file", filename)
        self.model = self._load(self.model, filename + ".a2c_critic")


A2C_Param = namedtuple('A2C_Param',
                       ['hidden_size',
                        'learning_rate',
                        'discount',
                        'clip_grad_norm',
                        'use_gpu'])


class A2CPolicy(LearningPolicy):
    def __init__(self,
                 state_size,
                 action_size,
                 in_parameters: Union[A2C_Param, None] = None):
        super(A2CPolicy, self).__init__()
        # parameters
        self.state_size = state_size
        self.action_size = action_size
        self.a2c_parameters = in_parameters
        if self.a2c_parameters is not None:
            self.hidden_size = self.a2c_parameters.hidden_size
            self.learning_rate = self.a2c_parameters.learning_rate
            self.discount = self.a2c_parameters.discount
            self.clip_grad_norm = self.a2c_parameters.clip_grad_norm

        else:
            self.hidden_size = 128
            self.learning_rate = 0.5e-3
            self.discount = 0.95
            self.clip_grad_norm = 0.1
            self.device = torch.device("cpu")

        # Device
        if self.a2c_parameters.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("🐇 Using GPU")
        else:
            self.device = torch.device("cpu")
            print("🐢 Using CPU")

        self.current_episode_memory = EpisodeBuffers()

        self.agent_done = {}
        self.memory = []  # dummy parameter

        self.loss_function = nn.MSELoss()
        self.loss = 0

        self.actor = A2CActor(state_size,
                              action_size,
                              self.device,
                              hidsize1=self.hidden_size,
                              hidsize2=self.hidden_size)
        self.critic = A2CCritic(state_size,
                                self.device,
                                hidsize1=self.hidden_size,
                                hidsize2=self.hidden_size)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

    def get_name(self):
        return self.__class__.__name__

    def shape_reward(self, handle, action, state, reward, done, deadlocked=None):
        return reward

    def reset(self, env):
        pass

    def act(self, handle, state, eps=0.0):
        torch_state = torch.tensor(state, dtype=torch.float).to(self.device)
        dist, _ = self.actor.get_actor_dist(torch_state)
        action = dist.sample()
        return action.item()

    def step(self, handle, state, action, reward, next_state, done):
        if self.agent_done.get(handle, False):
            return  # remove? if not Flatland?
        # record transitions ([state] -> [action] -> [reward, next_state, done])
        torch_action = torch.tensor(action, dtype=torch.float).to(self.device)
        torch_state = torch.tensor(state, dtype=torch.float).to(self.device)
        # evaluate actor
        dist, _ = self.actor.get_actor_dist(torch_state)
        action_logprobs = dist.log_prob(torch_action)
        transition = (state, action, reward, next_state, action_logprobs.item(), done)
        self.current_episode_memory.push_transition(handle, transition)
        if done:
            self.agent_done.update({handle: done})

    def _convert_transitions_to_torch_tensors(self, transitions_array, all_done):
        # build empty lists(arrays)
        state_list, action_list, reward_list, state_next_list, prob_a_list, done_list = [], [], [], [], [], []

        # set discounted_reward to zero
        discounted_reward = 0
        for transition in transitions_array[::-1]:
            state_i, action_i, reward_i, state_next_i, prob_action_i, done_i = transition
            state_list.insert(0, state_i)
            action_list.insert(0, action_i)
            done_list.insert(0, int(done_i))
            mask_i = 1.0 - int(done_i)
            discounted_reward = reward_i + self.discount * mask_i * discounted_reward
            reward_list.insert(0, discounted_reward)
            state_next_list.insert(0, state_next_i)
            prob_a_list.insert(0, prob_action_i)

        # convert data to torch tensors
        states, actions, rewards, states_next, dones, prob_actions = \
            torch.tensor(state_list, dtype=torch.float).to(self.device), \
                torch.tensor(action_list).to(self.device), \
                torch.tensor(reward_list, dtype=torch.float).to(self.device), \
                torch.tensor(state_next_list, dtype=torch.float).to(self.device), \
                torch.tensor(done_list, dtype=torch.float).to(self.device), \
                torch.tensor(prob_a_list).to(self.device)

        return states, actions, rewards, states_next, dones, prob_actions

    def train_net(self):
        # All agents have to propagate their experiences made during past episode
        all_done = True
        for handle in range(len(self.current_episode_memory)):
            all_done = self.agent_done.get(handle, False)

        for handle in range(len(self.current_episode_memory)):
            # Extract agent's episode history (list of all transitions)
            agent_episode_history = self.current_episode_memory.get_transitions(handle)
            if len(agent_episode_history) > 0:
                # Convert the replay buffer to torch tensors (arrays)
                states, actions, rewards, states_next, dones, probs_action = \
                    self._convert_transitions_to_torch_tensors(agent_episode_history, all_done)

                critic_loss = self.loss_function(rewards, torch.squeeze(self.critic(states)))
                self.optimizer_critic.zero_grad()
                critic_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.critic.model.parameters(), self.clip_grad_norm)
                self.optimizer_critic.step()

                advantages = rewards - torch.squeeze(self.critic(states)).detach()
                dist, _ = self.actor.get_actor_dist(states)
                action_logprobs = dist.log_prob(actions)
                self.optimizer_actor.zero_grad()
                actor_loss = -(advantages.detach() * action_logprobs) - 0.01 * dist.entropy()
                actor_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.model.parameters(), self.clip_grad_norm)
                self.optimizer_actor.step()

                # Transfer the current loss to the agents loss (information) for debug purpose only
                self.loss = actor_loss.mean().detach().cpu().numpy()

        # Reset all collect transition data
        self.current_episode_memory.reset()
        self.agent_done = {}

    def end_episode(self, train):
        if train:
            self.train_net()

    # Checkpointing methods
    def save(self, filename):
        # print("Saving model from checkpoint:", filename)
        self.actor.save(filename)
        self.critic.save(filename)
        torch.save(self.optimizer_actor.state_dict(), filename + ".a2c_optimizer_actor")
        torch.save(self.optimizer_critic.state_dict(), filename + ".a2c_optimizer_critic")

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            try:
                obj.load_state_dict(torch.load(filename, map_location=self.device))
            except:
                print(" >> failed!")
        else:
            print(" >> file not found!")
        return obj

    def load(self, filename):
        self.actor.load(filename)
        self.critic.load(filename)
        self.optimizer_actor = self._load(self.optimizer_actor, filename + ".a2c_optimizer_actor")
        self.optimizer_critic = self._load(self.optimizer_critic, filename + ".a2c_optimizer_critic")
        print('{} -> load {} ok'.format(self.get_name(), filename))

    def clone(self):
        policy = A2CPolicy(self.state_size, self.action_size)
        policy.actor = copy.deepcopy(self.actor)
        policy.critic = copy.deepcopy(self.critic)
        policy.optimizer_actor = copy.deepcopy(self.optimizer_actor)
        policy.optimizer_critic = copy.deepcopy(self.optimizer_critic)
        return policy
