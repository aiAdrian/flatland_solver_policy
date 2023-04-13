from collections import deque

import numpy as np
from flatland.envs.rail_env import RailEnv
from torch.utils.tensorboard import SummaryWriter

from solver.base_solver import BaseSolver


class FlatlandSolver(BaseSolver):
    def __init__(self, env: RailEnv):
        super(FlatlandSolver, self).__init__(env)
        self.policy = None

    def _run_step(self, env, policy, state, eps, training_mode):
        tot_reward = 0
        all_terminal = True

        policy.start_step(train=training_mode)

        actions = {}
        for handle in range(self.env.get_agent_handles()):
            policy.start_act(handle, train=training_mode)
            action = policy.act(handle, state[handle], eps)
            actions.update({handle: action})
            policy.end_act(handle, train=training_mode)

        state_next, reward, terminal, info = env.step(actions)
        for handle in range(self.env.get_agent_handles()):
            policy.step(handle, state[handle], actions[handle], reward[handle], state_next[handle], terminal[handle])
            all_terminal = all_terminal & terminal[handle]
            tot_reward += reward[handle]

        policy.end_step(train=training_mode)
        return state_next, tot_reward, all_terminal
