from solver.multi_agent_base_solver import MultiAgentBaseSolver


class MultiAgentGymSolver(MultiAgentBaseSolver):
    def get_name(self) -> str:
        return 'MAGym_{}'.format(self.env.get_name().replace(':', '_'))

    def transform_state(self, state):
        return state
