from flatland.core.env_observation_builder import ObservationBuilder
import numpy as np
import copy
from collections import deque
from typing import Optional, List, Dict
import os
import time
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
    3. Sequential Format: Returns [(obs_t-2, opp_t-2, tree_t-2), ...]
    """
    _last_obs_perf_report = None

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
                if base_obs == 'DecisionPointObservation':
                    self.base_obs = DecisionPointObservation()
                else:
                    self.base_obs = registry[base_obs]()
            else:
                raise ValueError(f"Unknown base_obs: {base_obs}")
        else:
            raise ValueError(f"Invalid base_obs: {base_obs}")
        self.env = None
        self.temporal_history: Dict[int, deque] = {}
        self.obs_time_profile_enabled = str(os.getenv('FLATLAND_OBS_TIME_PROFILE', '1')).strip().lower() in ('1', 'true', 'yes', 'on')
        self.obs_time_profile_interval = max(1, int(os.getenv('FLATLAND_OBS_TIME_PROFILE_INTERVAL_EPISODES', '20')))
        self._obs_time_total = 0.0
        self._obs_time_base = 0.0
        self._obs_time_calls = 0
        self._obs_last_report_episode = -1
        if not getattr(type(self), "_banner_printed", False):
            print(">> TemporalMultiAgentObservation loaded.")
            type(self)._banner_printed = True

    @staticmethod
    def _opponent_relevance_score(obs_vec: np.ndarray, tree_payload: Optional[Dict] = None) -> float:
        """Score opponent relevance from base features + raw local-tree payload.

        Base vector contribution:
        - [6:13] TrainState one-hot (activity signal)
        - [13]   priority rank (distance-based urgency)

        Tree payload contribution:
        - max deadlock risk over nodes
        - oncoming / incoming conflict cues
        - local branching pressure
        """
        if obs_vec is None:
            return 0.0

        v = np.asarray(obs_vec, dtype=np.float32).reshape(-1)
        state_activity = float(np.sum(v[6:13])) if v.shape[0] >= 13 else 0.0
        priority = float(v[13]) if v.shape[0] > 13 else 0.0

        payload = tree_payload if isinstance(tree_payload, dict) else {}
        nodes = payload.get("nodes", []) if isinstance(payload.get("nodes", []), list) else []

        max_deadlock = 0.0
        oncoming_ratio = 0.0
        branching_ratio = 0.0
        if len(nodes) > 0:
            max_deadlock = float(max(float(n.get("deadlock_risk", 0.0)) for n in nodes))
            oncoming_ratio = float(sum(1 for n in nodes if n.get("has_oncoming", False))) / float(len(nodes))
            branching_ratio = float(sum(1 for n in nodes if int(n.get("num_transitions", 0)) > 1)) / float(len(nodes))

        return (
            0.25 * state_activity
            + 0.35 * priority
            + 1.20 * max_deadlock
            + 0.90 * oncoming_ratio
            + 0.70 * branching_ratio
        )

    @staticmethod
    def getObservationSize() -> int:
        return DecisionPointObservation.getObservationSize()

    def get_observation_size(self) -> int:
        """Instance-level size that respects the wrapped base_obs (e.g. 24D for
        DecisionPointObservation or 48D for HierarchicalRoutesObservation)."""
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

    def get_many(self,
                 handles: Optional[List[int]] = None,
                 is_end_of_episode: bool = False,
                 episode_count: Optional[int] = None):
        t_total = time.perf_counter() if self.obs_time_profile_enabled else 0.0
        if handles is None:
            handles = list(range(len(self.env.agents)))

        # Forward optional episode-end flags to base builders that support them
        # (e.g. DecisionPointObservation summary/statistics every 100 episodes).
        try:
            t_base = time.perf_counter() if self.obs_time_profile_enabled else 0.0
            current_obs = self.base_obs.get_many(
                handles,
                is_end_of_episode=is_end_of_episode,
                episode_count=episode_count,
            )
            if self.obs_time_profile_enabled:
                self._obs_time_base += (time.perf_counter() - t_base)
        except TypeError:
            t_base = time.perf_counter() if self.obs_time_profile_enabled else 0.0
            current_obs = self.base_obs.get_many(handles)
            if self.obs_time_profile_enabled:
                self._obs_time_base += (time.perf_counter() - t_base)
        handle_to_obs = {h: obs_entry for h, obs_entry in zip(handles, current_obs)}

        def _unpack_base_obs(entry, handle):
            """Accept both legacy (obs, opps) and extended (obs, opps, payload)."""
            obs_vec = np.zeros(obs_size, dtype=np.float32)
            opp_handles = []
            payload = {}

            if isinstance(entry, (list, tuple)):
                if len(entry) >= 1:
                    obs_vec = np.asarray(entry[0], dtype=np.float32).reshape(-1)
                if len(entry) >= 2 and isinstance(entry[1], list):
                    opp_handles = entry[1]
                if len(entry) >= 3 and isinstance(entry[2], dict):
                    payload = entry[2]

            # DecisionPointObservation stores payload in env.dev_tree_dict.
            if not payload and hasattr(self.env, 'dev_tree_dict'):
                payload = self.env.dev_tree_dict.get(handle, {})

            return obs_vec, opp_handles, payload

        enriched_obs = []
        obs_size = self.get_observation_size()
        for handle, obs_entry in zip(handles, current_obs):
            obs_self, obs_others, tree_payload = _unpack_base_obs(obs_entry, handle)
            obs_fixed_size = obs_self[:obs_size]

            scored_opponents = []
            for opp_handle in obs_others:
                opp_entry = handle_to_obs.get(opp_handle, None)
                if opp_entry is None:
                    continue
                opp_base, _, _ = _unpack_base_obs(opp_entry, opp_handle)
                opp_base = opp_base[:obs_size]
                _, _, opp_payload = _unpack_base_obs(opp_entry, opp_handle)
                score = self._opponent_relevance_score(opp_base, opp_payload)
                scored_opponents.append((score, opp_base))

            if self.max_opponents > 0 and len(scored_opponents) > self.max_opponents:
                scored_opponents.sort(key=lambda x: x[0], reverse=True)
                scored_opponents = scored_opponents[:self.max_opponents]

            obs_others_enriched = [opp_base for _, opp_base in scored_opponents]
            enriched_obs.append((obs_fixed_size, obs_others_enriched, copy.deepcopy(tree_payload)))

        temporal_sequences = []
        for handle, (obs_self, obs_others, tree_payload) in zip(handles, enriched_obs):
            if handle not in self.temporal_history:
                self.temporal_history[handle] = deque(maxlen=self.temporal_window)
            self.temporal_history[handle].append((obs_self, obs_others, tree_payload))
            seq = list(self.temporal_history[handle])
            # Zero-pad early in the episode so the LSTM/temporal encoder
            # sees genuine velocity signals (replicating the first obs
            # produces zero deltas and biases the deadlock predictor).
            while len(seq) < self.temporal_window:
                zero_obs = np.zeros(obs_size, dtype=np.float32)
                seq.insert(0, (zero_obs, [], {}))
            temporal_sequences.append(seq)

        if self.obs_time_profile_enabled:
            self._obs_time_total += (time.perf_counter() - t_total)
            self._obs_time_calls += 1
            should_report = (
                is_end_of_episode
                and episode_count is not None
                and (episode_count + 1) % self.obs_time_profile_interval == 0
                and int(episode_count) != self._obs_last_report_episode
            )
            if should_report and self._obs_time_calls > 0:
                mean_total_ms = 1000.0 * self._obs_time_total / float(self._obs_time_calls)
                mean_base_ms = 1000.0 * self._obs_time_base / float(self._obs_time_calls)
                share_base = (100.0 * self._obs_time_base / self._obs_time_total) if self._obs_time_total > 1e-9 else 0.0
                print(
                    f"[ObsPerf] ep={episode_count + 1} interval={self.obs_time_profile_interval} "
                    f"calls={self._obs_time_calls} get_many_mean={mean_total_ms:.3f}ms "
                    f"base_obs_mean={mean_base_ms:.3f}ms base_share={share_base:.1f}%"
                )
                type(self)._last_obs_perf_report = {
                    'episode': int(episode_count + 1),
                    'interval': int(self.obs_time_profile_interval),
                    'calls': int(self._obs_time_calls),
                    'get_many_mean_ms': float(mean_total_ms),
                    'base_obs_mean_ms': float(mean_base_ms),
                    'base_share_pct': float(share_base),
                }
                self._obs_last_report_episode = int(episode_count)
                self._obs_time_total = 0.0
                self._obs_time_base = 0.0
                self._obs_time_calls = 0
        return temporal_sequences
