"""
spawn_aware_observation.py
==========================

SpawnAwareObservation extends DecisionPointObservation with 3 extra
features that capture global spawn / traffic state.

Why?
----
Disagreement analysis vs DLA showed that MAPPO never STOPs at OUTSIDE
(spawn zone), while DLA throttles 61% of the time. The base 22-dim
observation has no signal for "is the rail already crowded?" — so the
policy cannot learn spawn coordination.

Added features (3 dimensions appended → BASE_OBS_SIZE = 25):
  [22] active_density     — fraction of agents currently on the rail
  [23] pending_density    — fraction still waiting to depart
  [24] is_ready_to_depart — 1.0 if THIS agent is at READY_TO_DEPART

Usage
-----
    obs_builder = SpawnAwareObservation(debug=False, search_depth=4)
    # everything else (env, MAPPO) sees a 25-dim base feature vector

A/B testing
-----------
    Just swap the class:
        DecisionPointObservation(...)   # → 22 features (baseline)
        SpawnAwareObservation(...)      # → 25 features (with spawn awareness)
"""

from __future__ import annotations

import numpy as np

from .decision_point_observation import DecisionPointObservation


# Number of extra features this subclass appends
SPAWN_FEATURE_DIM = 3


class SpawnAwareObservation(DecisionPointObservation):
    """
    Adds 3 spawn / traffic features to the base DecisionPointObservation.

    The new BASE_OBS_SIZE is 25 (was 22). All downstream consumers should
    read `obs_builder.BASE_OBS_SIZE` rather than hard-coding 22.
    """

    # Override class constants so MAPPO can pick up the new dim
    BASE_OBS_SIZE = DecisionPointObservation.BASE_OBS_SIZE + SPAWN_FEATURE_DIM
    OBS_SIZE = BASE_OBS_SIZE

    # ─── FEATURE SPECS ─────────────────────────────────────────────────
    # Extend parent specs (22 dim) with 3 spawn-density features at indices
    # 22-24. Required by the BASE FEATURE REPORT to correctly map column
    # indices to feature names in the 25-dim observation vector.
    BASE_FEATURE_SPECS = list(DecisionPointObservation.BASE_FEATURE_SPECS) + [
        (22, "spawn_active_density",
         "[0,1] fraction of agents currently on the rail (not done)"),
        (23, "spawn_pending_density",
         "[0,1] fraction of agents WAITING / READY_TO_DEPART"),
        (24, "spawn_is_ready",
         "1 if THIS agent is READY_TO_DEPART/WAITING, else 0"),
    ]

    # Indices of the new features (for clarity / debugging)
    IDX_ACTIVE_DENSITY = 22
    IDX_PENDING_DENSITY = 23
    IDX_IS_READY = 24

    def __init__(self,
                 debug: bool = False,
                 search_depth: int = 4,
                 verbose_first_call: bool = False):
        super().__init__(debug=debug, search_depth=search_depth)
        self._spawn_first_call_logged = False
        self._spawn_verbose = bool(verbose_first_call)

    # --------------------------------------------------------------
    # Override _build_base_features: call parent then append 3 dims
    # --------------------------------------------------------------
    def _build_base_features(self, handle, agent, pos, direction, distance_map):
        # 1) Base 22 features from parent (parent uses self.BASE_OBS_SIZE,
        #    which we overrode to 25 → so we must call parent's logic on a
        #    22-dim array. Workaround: temporarily swap BASE_OBS_SIZE).
        original_size = SpawnAwareObservation.BASE_OBS_SIZE
        try:
            # Trick: set instance attribute so parent allocates 22 dims
            self.__dict__['BASE_OBS_SIZE'] = DecisionPointObservation.BASE_OBS_SIZE
            base22 = super()._build_base_features(
                handle, agent, pos, direction, distance_map
            )
        finally:
            # Restore: remove instance shadow so class attr (25) is visible
            self.__dict__.pop('BASE_OBS_SIZE', None)

        # 2) Compute 3 spawn features
        spawn_feats = self._compute_spawn_features(handle)

        # 3) Concatenate: 22 + 3 = 25
        full = np.concatenate([base22, spawn_feats]).astype(np.float32)

        if self._spawn_verbose and not self._spawn_first_call_logged:
            print(f"[SpawnAwareObservation] First base-features call: "
                  f"shape={full.shape}, "
                  f"spawn_feats={spawn_feats.tolist()} "
                  f"(active_density, pending_density, is_ready)")
            self._spawn_first_call_logged = True

        return full

    # --------------------------------------------------------------
    # The 3 new features
    # --------------------------------------------------------------
    def _compute_spawn_features(self, handle: int) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray of shape (3,):
            [active_density, pending_density, is_ready_to_depart]
        """
        agents = self.env.agents
        n_total = max(len(agents), 1)

        n_active = 0
        n_pending = 0
        for a in agents:
            pos = getattr(a, 'position', None)
            done = self._agent_is_done(a)
            state_name = self._agent_state_name(a)

            # Active = on the rail, not done
            if pos is not None and not done:
                n_active += 1
            # Pending = waiting to spawn
            if state_name in ('READY_TO_DEPART', 'WAITING'):
                n_pending += 1

        active_density = n_active / n_total
        pending_density = n_pending / n_total

        # Per-agent: am I myself ready to depart?
        is_ready = 0.0
        if 0 <= handle < len(agents):
            state_name = self._agent_state_name(agents[handle])
            if state_name in ('READY_TO_DEPART', 'WAITING'):
                is_ready = 1.0

        return np.array(
            [active_density, pending_density, is_ready],
            dtype=np.float32,
        )

    # --------------------------------------------------------------
    # Robust state inspection (Flatland version-agnostic)
    # --------------------------------------------------------------
    @staticmethod
    def _agent_state_name(agent) -> str:
        state = getattr(agent, 'state', None)
        if state is None:
            return 'UNKNOWN'
        name = getattr(state, 'name', None)
        if name:
            return str(name)
        return str(state)

    @staticmethod
    def _agent_is_done(agent) -> bool:
        if bool(getattr(agent, 'done', False)):
            return True
        state = getattr(agent, 'state', None)
        name = getattr(state, 'name', '') if state is not None else ''
        return name == 'DONE'
