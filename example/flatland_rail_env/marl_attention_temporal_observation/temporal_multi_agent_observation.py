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
    2. Emergent Velocity Learning: LSTM encoder learns temporal deltas from raw observation sequences
       (No explicit delta computation - LSTM infers motion from frame-to-frame changes)
    3. Sequential Format: Returns [(obs_t-2, opp_t-2), (obs_t-1, opp_t-1), (obs_t, opp_t)]
    """
    def __init__(self, temporal_window: int = 3, base_obs=None, max_opponents: int = 3):
        super().__init__()
        self.temporal_window = temporal_window
        self.max_opponents = max(0, int(max_opponents))
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
        if not getattr(type(self), "_banner_printed", False):
            print(">> TemporalMultiAgentObservation loaded.")
            type(self)._banner_printed = True

    @staticmethod
    def _opponent_relevance_score(obs_vec: np.ndarray) -> float:
        """
        Score opponent relevance using conflict-heavy features from
        DecisionPointObservation layout (48D). Falls back gracefully if shorter.
        """
        if obs_vec is None:
            return 0.0
        v = np.asarray(obs_vec, dtype=np.float32).reshape(-1)
        if v.shape[0] < 43:
            return float(np.linalg.norm(v, ord=1))

        decision_strength = float(v[0])
        branch_deadlock = float(v[5] + v[11] + v[17])
        merge_deadlock = float(v[22] + v[26])
        local_deadlock = float(v[42])
        coordination_wait = float(v[43]) if v.shape[0] > 43 else 0.0
        coordination_pressure = float(v[46]) if v.shape[0] > 46 else 0.0
        return (
            0.2 * decision_strength +
            1.0 * branch_deadlock +
            1.0 * merge_deadlock +
            1.5 * local_deadlock +
            0.6 * coordination_wait +
            0.6 * coordination_pressure
        )

    @staticmethod
    def getObservationSize() -> int:
        return DecisionPointObservation.getObservationSize()

    def get_observation_size(self) -> int:
        """Instance-level size that respects the wrapped base_obs (e.g. 72D for
        HierarchicalRoutesObservation)."""
        getter = getattr(self.base_obs, 'getObservationSize', None)
        if callable(getter):
            try:
                return int(getter())
            except Exception:
                pass
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
        handle_to_obs = {h: obs_pair for h, obs_pair in zip(handles, current_obs)}

        enriched_obs = []
        obs_size = self.get_observation_size()
        for handle, (obs_self, obs_others) in zip(handles, current_obs):
            obs_fixed_size = obs_self[:obs_size]

            scored_opponents = []
            for opp_handle in obs_others:
                opp_pair = handle_to_obs.get(opp_handle, None)
                if opp_pair is None:
                    continue
                opp_base = opp_pair[0][:obs_size]
                score = self._opponent_relevance_score(opp_base)
                scored_opponents.append((score, opp_base))

            if self.max_opponents > 0 and len(scored_opponents) > self.max_opponents:
                scored_opponents.sort(key=lambda x: x[0], reverse=True)
                scored_opponents = scored_opponents[:self.max_opponents]

            obs_others_enriched = [opp_base for _, opp_base in scored_opponents]
            enriched_obs.append((obs_fixed_size, obs_others_enriched))

        temporal_sequences = []
        for handle, (obs_self, obs_others) in zip(handles, enriched_obs):
            if handle not in self.temporal_history:
                self.temporal_history[handle] = deque(maxlen=self.temporal_window)
            self.temporal_history[handle].append((obs_self, obs_others))
            seq = list(self.temporal_history[handle])
            # Zero-pad early in the episode so the LSTM/temporal encoder
            # sees genuine velocity signals (replicating the first obs
            # produces zero deltas and biases the deadlock predictor).
            while len(seq) < self.temporal_window:
                zero_obs = np.zeros(obs_size, dtype=np.float32)
                seq.insert(0, (zero_obs, []))
            temporal_sequences.append(seq)
        return temporal_sequences
