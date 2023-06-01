from solver.flatland.flatland_solver import FlatlandSolver


class MultiAgentGymSolver(FlatlandSolver):
    def get_name(self) -> str:
        return 'MAGym_{}'.format(self.env.get_name().replace(':', '_'))

    def transform_state(self, state):
        return state
