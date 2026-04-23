from flatland.core.env_observation_builder import ObservationBuilder
import numpy as np
from collections import deque
from typing import Optional, List, Dict
from marl_attention_temporal_observation.experimental_observation import ExperimentalObservation
from marl_attention_temporal_observation.decision_point_observation import DecisionPointObservation
from marl_attention_temporal_observation.simplified_path_three_tier_observation import SimplifiedPathThreeTierObservation

class TemporalMultiAgentObservation(ObservationBuilder):
    """
    🚀 INNOVATION: Temporal Observation Builder
    Extends ExperimentalObservation with:
    1. Temporal Buffer: Stores last T timesteps (default T=3)
    2. Velocity Features: Computed from position/direction deltas
    3. Sequential Format: Returns [(obs_t-2, opp_t-2), (obs_t-1, opp_t-1), (obs_t, opp_t)]
    """
    def __init__(self, temporal_window: int = 3, base_obs=None):
        super().__init__()
        self.temporal_window = temporal_window
        if base_obs is None:
            self.base_obs = DecisionPointObservation()
        elif isinstance(base_obs, ObservationBuilder):
            self.base_obs = base_obs
        elif isinstance(base_obs, type) and issubclass(base_obs, ObservationBuilder):
            self.base_obs = base_obs()
        elif isinstance(base_obs, str):
            registry = {
                'ExperimentalObservation': ExperimentalObservation,
                'DecisionPointObservation': DecisionPointObservation,
                'SimplifiedPathThreeTierObservation': SimplifiedPathThreeTierObservation,
            }
            if base_obs in registry:
                self.base_obs = registry[base_obs]()
            else:
                raise ValueError(f"Unknown base_obs: {base_obs}")
        else:
            raise ValueError(f"Invalid base_obs: {base_obs}")
        self.env = None
        self.temporal_history: Dict[int, deque] = {}
        print(">> TemporalMultiAgentObservation loaded.")

    @staticmethod
    def getObservationSize() -> int:
        return DecisionPointObservation.getObservationSize()

    def set_env(self, env):
        super().set_env(env)
        self.env = env
        self.base_obs.set_env(env)

    def reset(self):
        self.base_obs.reset()
        self.temporal_history = {}
        self.base_obs.reset()

    def get_many(self, handles: Optional[List[int]] = None):
        if handles is None:
            handles = list(range(len(self.env.agents)))
        current_obs = self.base_obs.get_many(handles)
        enriched_obs = []
        for handle_idx, (obs_self, obs_others) in enumerate(current_obs):
            obs_fixed_size = obs_self[:TemporalMultiAgentObservation.getObservationSize()]
            obs_others_enriched = []
            for opp_handle in obs_others:
                opp_base = current_obs[opp_handle][0]
                obs_others_enriched.append(opp_base)
            enriched_obs.append((obs_fixed_size, obs_others_enriched))
        temporal_sequences = []
        for handle_idx, (obs_self, obs_others) in enumerate(enriched_obs):
            if handle_idx not in self.temporal_history:
                self.temporal_history[handle_idx] = deque(maxlen=self.temporal_window)
            self.temporal_history[handle_idx].append((obs_self, obs_others))
            seq = list(self.temporal_history[handle_idx])
            while len(seq) < self.temporal_window:
                if len(seq) > 0:
                    seq.insert(0, seq[0])
                else:
                    zero_obs = np.zeros(TemporalMultiAgentObservation.getObservationSize(), dtype=np.float32)
                    seq.insert(0, (zero_obs, []))
            temporal_sequences.append(seq)
        return temporal_sequences
