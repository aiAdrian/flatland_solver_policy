from flatland.envs.rail_env import RailEnv

from solver.base_renderer import BaseRenderer
from solver.base_solver import BaseSolver


class FlatlandSolver(BaseSolver):
    def __init__(self, env: RailEnv):
        super(FlatlandSolver, self).__init__(env)
        self.policy = None


    def get_name(self) -> str:
        return self.__class__.__name__


    def reset(self):
        state, _ = self.env.reset()
        if self.policy is not None:
            self.policy.reset(self.env)
        if self.renderer is not None:
            self.renderer.reset()
        return self.transform_state(state)

    def transform_state(self, state):
        return state

    def update_state(self, state_next):
        return state_next

    def run_step(self, env, policy, state, eps, training_mode):
        tot_reward = 0
        all_terminal = True

        policy.start_step(train=training_mode)

        actions = {}
        for handle in self.env.get_agent_handles():
            policy.start_act(handle, train=training_mode)
            action = policy.act(handle,
                                state[handle],
                                eps)
            actions.update({handle: action})
            policy.end_act(handle, train=training_mode)

        raw_state_next, reward, terminal, info = env.step(actions)
        state_next = self.transform_state(raw_state_next)

        for handle in self.env.get_agent_handles():
            policy.step(handle,
                        state[handle],
                        actions[handle],
                        reward[handle],
                        state_next[handle],
                        terminal[handle])
            all_terminal = all_terminal & terminal[handle]
            tot_reward += reward[handle]

        policy.end_step(train=training_mode)
        return state_next, tot_reward, all_terminal
