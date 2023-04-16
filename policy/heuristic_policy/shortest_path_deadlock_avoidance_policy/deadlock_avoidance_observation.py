from typing import Optional, List

import numpy as np
from flatland.core.env_observation_builder import DummyObservationBuilder


class DeadlockAvoidanceObservation(DummyObservationBuilder):
    def __init__(self):
        self.counter = 0

    def get_many(self, handles: Optional[List[int]] = None) -> bool:
        self.counter += 1
        obs = np.ones((len(handles), 2))
        for handle in handles:
            obs[handle][0] = handle
            obs[handle][1] = self.counter
        return obs
