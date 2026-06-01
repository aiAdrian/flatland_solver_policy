"""
ConflictAwareObservation — extends SpawnAwareObservation with CBS-style conflict features.

Hierarchy:
    DecisionPointObservation     (22 features)
        └── SpawnAwareObservation     (+3 spawn features = 25)
                └── ConflictAwareObservation     (+5 global +9 local +5 DLA = 44)

Provenance:
- Conflict-Based Search (CBS, Sharon et al. 2015)
- Cooperative A* (Silver 2005)
- PIBT / Priority-Based Search (Okumura et al. 2019)
- PRIMAL2 (Damani et al. 2021, arXiv:2010.08364)
"""

from typing import List, Optional, Tuple

import numpy as np

from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_argmax, fast_count_nonzero

from marl_attention_temporal_observation.spawn_aware_observation import SpawnAwareObservation
from marl_attention_temporal_observation.conflict_predictor import ConflictPredictor

from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.deadlock_avoidance_policy \
    import DeadLockAvoidancePolicy


class ConflictAwareObservation(SpawnAwareObservation):
    """Extends SpawnAwareObservation with 14 CBS-style conflict features + 5 DLA one-hot."""

    # ─── DIMENSIONS ──────────────────────────────────────────────────────
    BASE_OBS_SIZE = 44           # 25 (parent) + 5 (global) + 9 (local) + 5 (DLA)
    OBS_SIZE = BASE_OBS_SIZE
    GLOBAL_CONFLICT_DIMS = 5
    LOCAL_CONFLICT_DIMS = 9
    HORIZON_K = 5

    # ─── FEATURE SPECS ───────────────────────────────────────────────────
    BASE_FEATURE_SPECS = SpawnAwareObservation.BASE_FEATURE_SPECS + [
        (25, "global_conflict_pressure",
            "[0,1] fraction of vertex-cells with multi-agent reservations"),
        (26, "my_priority_global",
            "[0,1] my rank by remaining SP distance: 1.0=shortest"),
        (27, "my_sp_blocked_score",
            "[0,1] how much of my predicted SP is blocked by others"),
        (28, "sp_alternatives_avg_score",
            "[0,1] benefit of alternative branches"),
        (29, "expected_yield_count",
            "[0,1] fraction of higher-priority agents that benefit if I stop"),
        (30, "vertex_conflict_left",   "[0,1]"),
        (31, "edge_conflict_left",     "[0,1]"),
        (32, "lane_density_left",      "[0,1]"),
        (33, "vertex_conflict_forward","[0,1]"),
        (34, "edge_conflict_forward",  "[0,1]"),
        (35, "lane_density_forward",   "[0,1]"),
        (36, "vertex_conflict_right",  "[0,1]"),
        (37, "edge_conflict_right",    "[0,1]"),
        (38, "lane_density_right",     "[0,1]"),
        (39, "dla_action_noop",    "DLA one-hot"),
        (40, "dla_action_left",    "DLA one-hot"),
        (41, "dla_action_forward", "DLA one-hot"),
        (42, "dla_action_right",   "DLA one-hot"),
        (43, "dla_action_stop",    "DLA one-hot (also fallback)"),
    ]

    # ─── CONSTRUCTOR ─────────────────────────────────────────────────────
    def __init__(
        self,
        debug: bool = False,
        search_depth: int = 4,
        verbose_first_call: bool = True,
        conflict_horizon: int = HORIZON_K,
    ):
        super().__init__(
            debug=debug,
            search_depth=search_depth,
            verbose_first_call=verbose_first_call,
        )
        self._conflict_horizon = int(conflict_horizon)
        self._conflict_predictor: Optional[ConflictPredictor] = None
        self._conflict_first_call_logged = False
        self._conflict_verbose = bool(verbose_first_call)
        self._dla_cache: dict = {}
        self._internal_dla: Optional[DeadLockAvoidancePolicy] = None
        self._internal_dla_env_id: Optional[int] = None

    # ─── LIFECYCLE ───────────────────────────────────────────────────────
    def reset(self):
        super().reset()
        self._conflict_predictor = None
        self._conflict_first_call_logged = False
        self._dla_cache = {}
        # DLA reset is handled lazily in get_many() (env may not be ready yet here)

    def get_many(
        self,
        handles: list = None,
        is_end_of_episode: bool = False,
        episode_count: int = None,
    ):
        """Init/refresh predictor + DLA, then delegate to parent."""
        # Lazy init predictor
        if self._conflict_predictor is None and self.env is not None:
            self._conflict_predictor = ConflictPredictor(
                self.env, horizon=self._conflict_horizon
            )

        # Build agent_map and update predictor
        if self.env is not None:
            agent_map = np.full(
                (self.env.height, self.env.width), -1, dtype=np.int32
            )
            for agent in self.env.agents:
                if agent.position is not None:
                    agent_map[agent.position] = agent.handle

            if self._conflict_predictor is not None:
                self._conflict_predictor.update(agent_map)

        # DLA: re-instantiate when n_agents changes. The persister reuses
        # env objects (id() stays constant), so we must check n_agents directly.
        # If start_step fails, drop the DLA so next call rebuilds fresh.
        if self.env is not None:
            n_agents_now = len(self.env.agents)
            need_reinit = (
                self._internal_dla is None
                or getattr(self._internal_dla, "_cached_n_agents", -1) != n_agents_now
            )
            if need_reinit:
                try:
                    self._internal_dla = DeadLockAvoidancePolicy(
                        self.env, action_size=5, enable_eps=False
                    )
                    self._internal_dla._cached_n_agents = n_agents_now
                    self._internal_dla_env_id = id(self.env)
                except Exception as e:
                    print(f"[ConflictAwareObs] DLA init failed: {e}")
                    self._internal_dla = None
                    self._internal_dla_env_id = None
            if self._internal_dla is not None:
                try:
                    self._internal_dla.start_step(train=False)
                    self._dla_cache = self._internal_dla.agent_can_move or {}
                except Exception as e:
                    # Drop DLA so it rebuilds fresh next call
                    print(f"[ConflictAwareObs] DLA start_step failed: {e} "
                          f"(n_agents={n_agents_now}) — dropping for fresh init")
                    self._internal_dla = None
                    self._dla_cache = {}
        return super().get_many(handles, is_end_of_episode, episode_count)

    # ─── DLA HELPERS ─────────────────────────────────────────────────────
    def set_dla_cache(self, dla_agent_can_move):
        """External setter (optional) — call before get_many()."""
        self._dla_cache = dla_agent_can_move or {}

    def _build_dla_features(self, handle):
        """5-dim one-hot of DLA-recommended action; STOP if unknown."""
        feats = np.zeros(5, dtype=np.float32)
        entry = self._dla_cache.get(handle, None)
        if entry is None:
            feats[4] = 1.0  # default: STOP
            return feats
        try:
            action_val = entry[3] if isinstance(entry, (tuple, list)) else entry
            action_int = int(action_val)
            if 0 <= action_int <= 4:
                feats[action_int] = 1.0
            else:
                feats[4] = 1.0
        except (IndexError, TypeError, ValueError):
            feats[4] = 1.0
        return feats

    # ─── FEATURE BUILDER (override) ──────────────────────────────────────
    def _build_base_features(self, handle, agent, pos, direction, distance_map):
        """Concat: 25 base + 5 global + 9 local + 5 DLA = 44."""
        base_25 = super()._build_base_features(handle, agent, pos, direction, distance_map)
        global_5 = self._build_global_conflict_features(handle, agent, pos, direction, distance_map)
        local_9 = self._build_local_conflict_features(handle, agent, pos, direction, distance_map)
        dla_5 = self._build_dla_features(handle)
        full = np.concatenate([base_25, global_5, local_9, dla_5]).astype(np.float32)

        assert full.shape[0] == self.BASE_OBS_SIZE, (
            f"ConflictAwareObservation size mismatch: "
            f"got {full.shape[0]}, expected {self.BASE_OBS_SIZE}"
        )

        if self._conflict_verbose and not self._conflict_first_call_logged:
            print(
                f"[ConflictAwareObservation] First _build_base_features call:\n"
                f"  shape    = {full.shape}\n"
                f"  base[:5] = {base_25.tolist()[:5]}...\n"
                f"  global   = {global_5.tolist()}\n"
                f"  local    = {local_9.tolist()}\n"
                f"  dla(1h)  = {dla_5.tolist()}"
            )
            self._conflict_first_call_logged = True

        return full

    # ─── GLOBAL FEATURE BUILDERS ─────────────────────────────────────────
    def _build_global_conflict_features(
        self, handle, agent, pos, direction, distance_map
    ) -> np.ndarray:
        feats = np.zeros(5, dtype=np.float32)
        if self._conflict_predictor is None:
            return feats
        feats[0] = self._conflict_predictor.global_conflict_pressure()
        feats[1] = self._conflict_predictor.my_priority_global(handle)
        feats[2] = self._conflict_predictor.my_sp_blocked_score(handle)
        feats[3] = self._compute_sp_alternatives_score(handle, pos, direction, distance_map)
        feats[4] = self._conflict_predictor.expected_yield_count(handle)
        return feats

    def _compute_sp_alternatives_score(
        self, handle, pos, direction, distance_map
    ) -> float:
        if pos is None or direction is None or self._conflict_predictor is None:
            return 0.0

        my_sp_blocked = self._conflict_predictor.my_sp_blocked_score(handle)
        if my_sp_blocked < 0.05:
            return 0.0

        transitions = self._rail_get_transitions(pos, direction)
        if fast_count_nonzero(transitions) <= 1:
            return 0.0

        sp_best_dir = None
        sp_best_dist = float("inf")
        for nd in range(4):
            if not transitions[nd]:
                continue
            npos = get_new_position(pos, nd)
            if (npos[0] < 0 or npos[0] >= self.env.height
                    or npos[1] < 0 or npos[1] >= self.env.width):
                continue
            d = float(distance_map[handle, npos[0], npos[1], nd])
            if d < sp_best_dist:
                sp_best_dist = d
                sp_best_dir = nd

        if sp_best_dir is None:
            return 0.0

        alt_blocked_scores = []
        for nd in range(4):
            if not transitions[nd] or nd == sp_best_dir:
                continue
            alt_path = self._project_short_path(
                pos, nd, handle, distance_map, max_len=self._conflict_horizon
            )
            if len(alt_path) <= 1:
                continue

            n_blocked = 0
            for t, alt_pos in enumerate(alt_path[1:], start=1):
                users = self._conflict_predictor._vertex_res.get(
                    (alt_pos[0], alt_pos[1], t), []
                )
                others = [h for h in users if h != handle]
                if others:
                    n_blocked += 1
            alt_score = n_blocked / max(1, len(alt_path) - 1)
            alt_blocked_scores.append(alt_score)

        if not alt_blocked_scores:
            return 0.0

        avg_alt_blocked = float(np.mean(alt_blocked_scores))
        improvement = my_sp_blocked - avg_alt_blocked
        return float(np.clip(improvement, 0.0, 1.0))

    def _project_short_path(
        self, start_pos, start_dir, handle, distance_map, max_len=5
    ) -> List[Tuple[int, int]]:
        path = [(int(start_pos[0]), int(start_pos[1]))]

        first_npos = get_new_position(start_pos, start_dir)
        if (first_npos[0] < 0 or first_npos[0] >= self.env.height
                or first_npos[1] < 0 or first_npos[1] >= self.env.width):
            return path
        path.append((int(first_npos[0]), int(first_npos[1])))

        cur_pos = (int(first_npos[0]), int(first_npos[1]))
        cur_dir = int(start_dir)

        for _ in range(max_len - 1):
            transitions = self.env.rail.get_transitions(cur_pos[0], cur_pos[1], cur_dir)
            n_trans = fast_count_nonzero(transitions)
            if n_trans == 0:
                break

            best_dir = None
            best_dist = float("inf")
            for nd in range(4):
                if not transitions[nd]:
                    continue
                npos = get_new_position(cur_pos, nd)
                if (npos[0] < 0 or npos[0] >= self.env.height
                        or npos[1] < 0 or npos[1] >= self.env.width):
                    continue
                d = float(distance_map[handle, npos[0], npos[1], nd])
                if d < best_dist:
                    best_dist = d
                    best_dir = nd

            if best_dir is None:
                break

            next_pos = get_new_position(cur_pos, best_dir)
            path.append((int(next_pos[0]), int(next_pos[1])))
            cur_pos = (int(next_pos[0]), int(next_pos[1]))
            cur_dir = int(best_dir)

        return path

    # ─── LOCAL FEATURE BUILDERS ──────────────────────────────────────────
    def _build_local_conflict_features(
        self, handle, agent, pos, direction, distance_map
    ) -> np.ndarray:
        """9-dim: [vh_L, edge_L, lane_L, vh_F, edge_F, lane_F, vh_R, edge_R, lane_R]."""
        feats = np.zeros(9, dtype=np.float32)

        if (self._conflict_predictor is None or pos is None or direction is None):
            return feats

        transitions = self._rail_get_transitions(pos, direction)
        left_dir = (int(direction) - 1) % 4
        fwd_dir = int(direction) % 4
        right_dir = (int(direction) + 1) % 4

        for slot, ndir in enumerate((left_dir, fwd_dir, right_dir)):
            base = slot * 3
            if not transitions[ndir]:
                continue

            target_cell = get_new_position(pos, ndir)
            if (target_cell[0] < 0 or target_cell[0] >= self.env.height
                    or target_cell[1] < 0 or target_cell[1] >= self.env.width):
                continue

            feats[base + 0] = self._conflict_predictor.vertex_conflict_horizon(
                target_cell, handle
            )
            feats[base + 1] = self._conflict_predictor.edge_conflict(
                pos, target_cell, handle
            )
            corridor = self._collect_corridor_cells(target_cell, ndir, max_len=6)
            n_others = self._conflict_predictor.lane_agent_count(corridor, handle)
            feats[base + 2] = float(np.clip(n_others / 3.0, 0.0, 1.0))

        return feats

    def _collect_corridor_cells(
        self, start_cell, direction, max_len=6
    ) -> List[Tuple[int, int]]:
        cells = [(int(start_cell[0]), int(start_cell[1]))]
        p = (int(start_cell[0]), int(start_cell[1]))
        d = int(direction)

        for _ in range(max_len - 1):
            trans = self._rail_get_transitions(p, d)
            n_trans = fast_count_nonzero(trans)
            if n_trans != 1:
                break
            d = int(fast_argmax(trans))
            p = get_new_position(p, d)
            if (p[0] < 0 or p[0] >= self.env.height
                    or p[1] < 0 or p[1] >= self.env.width):
                break
            cells.append((int(p[0]), int(p[1])))

        return cells

    # ─── TREE EDGE ENRICHMENT (override) ─────────────────────────────────
    def _build_corridor_edge_payload(
        self, handle, src_idx, dst_idx, src_pos, src_dir, src_depth,
        dst_pos, dst_dir, corridor, distance_map,
    ):
        """Add conflict-aware fields to each tree edge."""
        edge = super()._build_corridor_edge_payload(
            handle=handle, src_idx=src_idx, dst_idx=dst_idx,
            src_pos=src_pos, src_dir=src_dir, src_depth=src_depth,
            dst_pos=dst_pos, dst_dir=dst_dir,
            corridor=corridor, distance_map=distance_map,
        )

        edge_cells = edge.get("cells", [])
        edge_agents = edge.get("agents", [])

        # conflict_density
        if self._conflict_predictor is not None and edge_cells:
            n_conflict_cells = 0
            for cell in edge_cells:
                conflict_found = False
                for t in range(1, self._conflict_horizon + 1):
                    users = self._conflict_predictor._vertex_res.get(
                        (int(cell[0]), int(cell[1]), t), []
                    )
                    if any(h != handle for h in users):
                        conflict_found = True
                        break
                if conflict_found:
                    n_conflict_cells += 1
            edge["conflict_density"] = float(n_conflict_cells / max(1, len(edge_cells)))
        else:
            edge["conflict_density"] = 0.0

        # priority_yield
        if self._conflict_predictor is not None and edge_agents:
            my_d = self._conflict_predictor._sp_distance_at_start.get(
                handle, float("inf")
            )
            n_lower_priority = 0
            for other_handle in edge_agents:
                other_d = self._conflict_predictor._sp_distance_at_start.get(
                    other_handle, float("inf")
                )
                if np.isfinite(other_d) and np.isfinite(my_d) and other_d > my_d:
                    n_lower_priority += 1
            edge["priority_yield"] = float(
                n_lower_priority / max(1, len(edge_agents))
            )
        else:
            edge["priority_yield"] = 0.0

        # sp_dominance
        if self._conflict_predictor is not None and edge_agents:
            dst_dist = edge.get("dst_dist_to_target")
            if dst_dist is not None and np.isfinite(dst_dist):
                others_dists = []
                for other_handle in edge_agents:
                    other_d = self._conflict_predictor._sp_distance_at_start.get(
                        other_handle, float("inf")
                    )
                    if np.isfinite(other_d):
                        others_dists.append(other_d)
                if others_dists:
                    max_other = max(others_dists)
                    if max_other > 0:
                        edge["sp_dominance"] = float(
                            np.clip(1.0 - dst_dist / max_other, 0.0, 1.0)
                        )
                    else:
                        edge["sp_dominance"] = 0.0
                else:
                    edge["sp_dominance"] = 0.0
            else:
                edge["sp_dominance"] = 0.0
        else:
            edge["sp_dominance"] = 0.0

        return edge

