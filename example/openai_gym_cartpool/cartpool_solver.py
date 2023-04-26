from solver.base_solver import BaseSolver


class CartPoolSolver(BaseSolver):
    def get_name(self) -> str:
        return self.__class__.__name__
