from policy.policy import Policy


class CartPoolAnalyticalPolicy(Policy):
    def get_name(self) -> str:
        return self.__class__.__name__

    def act(self, handle: int, state, eps=0.):
        t, w = state[2:4]
        if abs(t) < 0.03:
            return 0 if w < 0 else 1
        return 0 if t < 0 else 1

    def step(self, handle: int, state, action, reward, next_state, done):
        pass
