from typing import Union

from environment.environment import Environment
from policy.policy import Policy
from rendering.base_renderer import BaseRenderer
from solver.multi_agent_base_solver import MultiAgentBaseSolver


class FlatlandSolver(MultiAgentBaseSolver):
    def __init__(self,
                 env: Environment,
                 policy: Policy,
                 renderer: Union[BaseRenderer, None] = None):
        super(FlatlandSolver, self).__init__(env, policy, renderer)

    def get_name(self) -> str:
        return self.__class__.__name__

    def run_choose_action(self, eps, handle, info, policy, state):
        if info['action_required'][handle]:
            updated = True
            action = policy.act(handle,
                                state[handle],
                                eps)
        else:
            # An action is not required if the train hasn't joined the railway network,
            # if it already reached its target, or if the train is currently malfunctioning.
            updated = False
            action = 0
        return action, updated
