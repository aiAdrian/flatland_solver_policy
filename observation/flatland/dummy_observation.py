from typing import Optional, List

import numpy as np
from flatland.core.env_observation_builder import DummyObservationBuilder


class FlatlandDummyObservation(DummyObservationBuilder):
    def __init__(self):
        pass

    def get_many(self, handles: Optional[List[int]] = None) -> bool:
        obs = np.ones((len(handles), 1))
        for handle in handles:
            obs[handle][0] = handle
        return obs
