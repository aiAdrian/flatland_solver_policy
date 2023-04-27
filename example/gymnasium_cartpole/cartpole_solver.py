from solver.base_solver import BaseSolver


class CartpoleSolver(BaseSolver):
    def get_name(self) -> str:
        return self.__class__.__name__
