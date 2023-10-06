import numpy as np

from environment.environment import Environment
from policy.heuristic_policy.heuristic_policy import HeuristicPolicy
from policy.learning_policy.learning_policy import LearningPolicy
from policy.policy import Policy


class ReinforceHeuristicPolicy(LearningPolicy):

    def __init__(self,
                 learning_policy: LearningPolicy,
                 heuristic_policy: HeuristicPolicy,
                 heuristic_policy_epsilon=0.1):
        super(ReinforceHeuristicPolicy, self).__init__()
        self.learning_policy: Policy = learning_policy
        self.heuristic_policy: Policy = heuristic_policy
        self.heuristic_policy_epsilon = heuristic_policy_epsilon

    def get_name(self):
        return self.__class__.__name__

    def save(self, filename):
        self.learning_policy.save(filename)
        self.heuristic_policy.save(filename)

    def load(self, filename):
        self.learning_policy.load(filename)
        self.heuristic_policy.load(filename)

    def start_episode(self, train: bool):
        self.learning_policy.start_episode(train)
        self.heuristic_policy.start_episode(train)

    def start_step(self, train: bool):
        self.learning_policy.start_step(train)
        self.heuristic_policy.start_step(train)

    def act(self, handle: int, state, eps=0.):
        # Epsilon-greedy action selection
        if np.random.random() >= (eps * max(0.0, min(1.0, (1.0 - self.heuristic_policy_epsilon)))):
            return self.learning_policy.act(handle, state, eps)
        else:
            return self.heuristic_policy.act(handle, state, eps)

    def step(self, handle: int, state, action, reward, next_state, done):
        self.learning_policy.step(handle, state, action, reward, next_state, done)
        self.heuristic_policy.step(handle, state, action, reward, next_state, done)

    def end_step(self, train: bool):
        self.learning_policy.end_step(train)
        self.heuristic_policy.end_step(train)

    def end_episode(self, train: bool):
        self.learning_policy.end_episode(train)
        self.heuristic_policy.end_episode(train)

    def load_replay_buffer(self, filename):
        self.learning_policy.load_replay_buffer(filename)
        self.heuristic_policy.load_replay_buffer(filename)

    def test(self):
        self.learning_policy.test()
        self.heuristic_policy.test()

    def reset(self, env: Environment):
        self.learning_policy.reset(env)
        self.heuristic_policy.reset(env)

    def clone(self):
        lp = self.learning_policy.clone()
        hp = self.heuristic_policy.clone()
        return ReinforceHeuristicPolicy(lp, hp)
