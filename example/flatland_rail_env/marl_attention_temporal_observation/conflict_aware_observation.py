"""
ConflictAwareObservation — extends SpawnAwareObservation with CBS-style conflict features.

Hierarchy:
    DecisionPointObservation     (22 features)
        └── SpawnAwareObservation     (+3 spawn features = 25)
                └── ConflictAwareObservation     (+5 global +9 local = 39)   ← this file

Provenance:
- Conflict-Based Search (CBS, Sharon et al. 2015) — pairwise conflict detection
- Cooperative A* (Silver 2005) — time-space reservation tables
- PIBT / Priority-Based Search (Okumura et al. 2019) — priority hierarchy for yielding
- PRIMAL2 (Damani et al. 2021, arXiv:2010.08364) — local conflict awareness for MARL

Key design ideas:
1. GLOBAL features (5):
   - Tell the agent how busy the system is and where it stands in the
     priority hierarchy.
2. LOCAL features per direction (3 × 3 = 9):
   - Per-action conflict information; tells the policy WHICH action
     causes/avoids conflict.
3. TREE EDGE ENRICHMENT (no new dims):
   - Each tree edge gets conflict_density, priority_yield_signal,
     so the existing Tree Encoder gets richer context.
"""

from typing import List, Optional, Tuple

import numpy as np

from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_argmax, fast_count_nonzero
from flatland.envs.step_utils.states import TrainState

from marl_attention_temporal_observation.spawn_aware_observation import SpawnAwareObservation
from marl_attention_temporal_observation.conflict_predictor import ConflictPredictor


class ConflictAwareObservation(SpawnAwareObservation):
    """Extends SpawnAwareObservation with 14 CBS-style conflict features."""

    # ─── DIMENSIONS ──────────────────────────────────────────────────────
    BASE_OBS_SIZE = 39           # 25 (parent) + 5 (global) + 9 (local)
    OBS_SIZE = BASE_OBS_SIZE
    GLOBAL_CONFLICT_DIMS = 5
    LOCAL_CONFLICT_DIMS = 9      # 3 directions × 3 features
    HORIZON_K = 5                # trajectory prediction depth

    # ─── FEATURE SPECS ───────────────────────────────────────────────────
    # We extend the parent's feature spec list with 14 new entries.
    # The first 25 indices are inherited from SpawnAwareObservation.
    BASE_FEATURE_SPECS = SpawnAwareObservation.BASE_FEATURE_SPECS + [
        (25, "global_conflict_pressure",
            "[0,1] fraction of vertex-cells with multi-agent reservations"),
        (26, "my_priority_global",
            "[0,1] my rank by remaining SP distance: 1.0=shortest dist (move!)"),
        (27, "my_sp_blocked_score",
            "[0,1] how much of my predicted SP is blocked by others"),
        (28, "sp_alternatives_avg_score",
            "[0,1] how much would alternative branches help avoid conflict"),
        (29, "expected_yield_count",
            "[0,1] fraction of higher-priority agents that benefit if I stop"),
        (30, "vertex_conflict_left",
            "[0,1] when another agent enters my left-target (1=immediate)"),
        (31, "edge_conflict_left",
            "[0,1] head-on swap conflict for left direction"),
        (32, "lane_density_left",
            "[0,1] number of other agents in left corridor (norm by 3)"),
        (33, "vertex_conflict_forward",
            "[0,1] when another agent enters my forward-target"),
        (34, "edge_conflict_forward",
            "[0,1] head-on swap conflict for forward direction"),
        (35, "lane_density_forward",
            "[0,1] number of other agents in forward corridor"),
        (36, "vertex_conflict_right",
            "[0,1] when another agent enters my right-target"),
        (37, "edge_conflict_right",
            "[0,1] head-on swap conflict for right direction"),
        (38, "lane_density_right",
            "[0,1] number of other agents in right corridor"),
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

    # ─── LIFECYCLE ───────────────────────────────────────────────────────
    def reset(self):
        super().reset()
        # Predictor lifecycle is bound to env-step, not to episode reset.
        # We allocate it lazily on first get_many() call (env must be set).
        self._conflict_predictor = None
        self._conflict_first_call_logged = False

    def get_many(
        self,
        handles: list = None,
        is_end_of_episode: bool = False,
        episode_count: int = None,
    ):
        """Override: ensure predictor is initialized & updated before parent runs.
        
        Important: parent's get_many() does TWO things we depend on:
          1. Builds self.agent_map (needed for predictor)
          2. Calls self.get(handle) → which calls _build_base_features
        
        We must run our predictor.update() AFTER step 1 but BEFORE step 2.
        Strategy: we replicate the agent_map construction (cheap) and update
        the predictor BEFORE calling super().get_many().
        """
        # Lazy init
        if self._conflict_predictor is None and self.env is not None:
            self._conflict_predictor = ConflictPredictor(
                self.env, horizon=self._conflict_horizon
            )

        # Pre-build agent_map (parent will rebuild it, that's fine; it's cheap)
        if self.env is not None:
            agent_map = np.zeros(
                (self.env.height, self.env.width), dtype=np.int32
            ) - 1
            for agent in self.env.agents:
                if agent.position is not None:
                    agent_map[agent.position] = agent.handle

            # Update predictor with current world state
            if self._conflict_predictor is not None:
                self._conflict_predictor.update(agent_map)

        # Now delegate to parent — its get() will call our overridden
        # _build_base_features which uses self._conflict_predictor.
        return super().get_many(handles, is_end_of_episode, episode_count)

    # ─── FEATURE BUILDER (override) ──────────────────────────────────────
    def _build_base_features(self, handle, agent, pos, direction, distance_map):
        """Concatenate parent's 25 features with 5 global + 9 local conflict features."""
        # 1. Get 25-dim base from parent (DecisionPoint + spawn)
        base_25 = super()._build_base_features(handle, agent, pos, direction, distance_map)

        # 2. Compute 5-dim global conflict features
        global_5 = self._build_global_conflict_features(
            handle, agent, pos, direction, distance_map
        )

        # 3. Compute 9-dim local conflict features
        local_9 = self._build_local_conflict_features(
            handle, agent, pos, direction, distance_map
        )

        # 4. Concatenate
        full = np.concatenate([base_25, global_5, local_9]).astype(np.float32)

        assert full.shape[0] == self.BASE_OBS_SIZE, (
            f"ConflictAwareObservation size mismatch: "
            f"got {full.shape[0]}, expected {self.BASE_OBS_SIZE}"
        )

        # 5. Debug-log on first call
        if self._conflict_verbose and not self._conflict_first_call_logged:
            print(
                f"[ConflictAwareObservation] First _build_base_features call:\n"
                f"  shape           = {full.shape}\n"
                f"  base (25-dim)   = {base_25.tolist()[:5]}... (truncated)\n"
                f"  global (5-dim)  = {global_5.tolist()}\n"
                f"  local (9-dim)   = {local_9.tolist()}\n"
                f"  meaning(global) = [conflict_pressure, my_priority, sp_blocked, "
                f"alt_score, yield_count]\n"
                f"  meaning(local)  = [vh_L, edge_L, lane_L, vh_F, edge_F, lane_F, "
                f"vh_R, edge_R, lane_R]"
            )
            self._conflict_first_call_logged = True

        return full

    # ─── GLOBAL FEATURE BUILDERS ─────────────────────────────────────────
    def _build_global_conflict_features(
        self, handle, agent, pos, direction, distance_map
    ) -> np.ndarray:
        """Return 5-dim vector with system-wide conflict context."""
        feats = np.zeros(5, dtype=np.float32)

        if self._conflict_predictor is None:
            return feats

        # [25] global_conflict_pressure
        feats[0] = self._conflict_predictor.global_conflict_pressure()

        # [26] my_priority_global
        feats[1] = self._conflict_predictor.my_priority_global(handle)

        # [27] my_sp_blocked_score
        feats[2] = self._conflict_predictor.my_sp_blocked_score(handle)

        # [28] sp_alternatives_avg_score — NOT YET in predictor; compute here
        feats[3] = self._compute_sp_alternatives_score(
            handle, pos, direction, distance_map
        )

        # [29] expected_yield_count
        feats[4] = self._conflict_predictor.expected_yield_count(handle)

        return feats

    def _compute_sp_alternatives_score(
        self, handle, pos, direction, distance_map
    ) -> float:
        """How much do alternative branches help avoid conflicts?
        
        Compares:
          - my SP-blocked score (using my actual predicted path)
          - mean SP-blocked score if I took a non-SP transition instead
        
        Returns [0, 1]:
          - 0.0 → no alternatives or alternatives equally blocked
          - 1.0 → alternatives are clearly better than SP
        """
        if pos is None or direction is None or self._conflict_predictor is None:
            return 0.0

        my_sp_blocked = self._conflict_predictor.my_sp_blocked_score(handle)
        if my_sp_blocked < 0.05:
            # SP is already clear → no benefit from alternatives
            return 0.0

        # Find non-SP transitions and estimate their blocked-ness
        transitions = self._rail_get_transitions(pos, direction)
        if fast_count_nonzero(transitions) <= 1:
            # No alternative branches available
            return 0.0

        # Find SP-best direction (smallest distance)
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

        # For each non-SP transition: walk a few steps along it, count
        # other agents predicted to overlap.
        alt_blocked_scores = []
        for nd in range(4):
            if not transitions[nd] or nd == sp_best_dir:
                continue

            # Project a short path along this alternative
            alt_path = self._project_short_path(
                pos, nd, handle, distance_map, max_len=self._conflict_horizon
            )
            if len(alt_path) <= 1:
                continue

            # Count overlap with others' predicted paths
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
        # Improvement = how much better alternatives are vs my SP
        improvement = my_sp_blocked - avg_alt_blocked
        return float(np.clip(improvement, 0.0, 1.0))

    def _project_short_path(
        self, start_pos, start_dir, handle, distance_map, max_len=5
    ) -> List[Tuple[int, int]]:
        """Project a short path from start_pos along start_dir using SP-greedy choice.
        
        Used by _compute_sp_alternatives_score. Same logic as
        ConflictPredictor._project_path but starting from a chosen direction.
        """
        path = [(int(start_pos[0]), int(start_pos[1]))]

        # First step: take the chosen direction explicitly
        first_npos = get_new_position(start_pos, start_dir)
        if (first_npos[0] < 0 or first_npos[0] >= self.env.height
                or first_npos[1] < 0 or first_npos[1] >= self.env.width):
            return path
        path.append((int(first_npos[0]), int(first_npos[1])))

        cur_pos = (int(first_npos[0]), int(first_npos[1]))
        cur_dir = int(start_dir)

        # Continue greedy SP for remaining steps
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
        """Return 9-dim vector with per-direction conflict info.
        
        Layout:
          [vh_L, edge_L, lane_L,  vh_F, edge_F, lane_F,  vh_R, edge_R, lane_R]
        
        For each direction L/F/R:
          - vh: vertex_conflict_horizon (0..1, 1=immediate)
          - edge: head-on swap conflict (0/1)
          - lane: corridor density (0..1, normalized by 3 agents)
        """
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
                continue  # transition not legal → all zeros for this slot

            target_cell = get_new_position(pos, ndir)
            # Bounds check
            if (target_cell[0] < 0 or target_cell[0] >= self.env.height
                    or target_cell[1] < 0 or target_cell[1] >= self.env.width):
                continue

            # Feature 0: vertex_conflict_horizon
            feats[base + 0] = self._conflict_predictor.vertex_conflict_horizon(
                target_cell, handle
            )

            # Feature 1: edge_conflict (head-on swap)
            feats[base + 1] = self._conflict_predictor.edge_conflict(
                pos, target_cell, handle
            )

            # Feature 2: lane_density
            corridor = self._collect_corridor_cells(target_cell, ndir, max_len=6)
            n_others = self._conflict_predictor.lane_agent_count(corridor, handle)
            feats[base + 2] = float(np.clip(n_others / 3.0, 0.0, 1.0))

        return feats

    def _collect_corridor_cells(
        self, start_cell, direction, max_len=6
    ) -> List[Tuple[int, int]]:
        """Walk forward through a corridor (≤1 transitions per cell), collect cells.
        
        Used by lane_density: gives the predictor a list of cells to scan for
        other agents currently occupying the same lane.
        
        Args:
            start_cell: (r, c) — first cell of the corridor (in-bounds)
            direction:  int 0..3 — heading we entered start_cell with
            max_len:    int — how many cells to collect at most
        
        Returns:
            List of (r, c) tuples, length 1..max_len. Stops early at switches,
            out-of-bounds, or dead ends.
        """
        cells = [(int(start_cell[0]), int(start_cell[1]))]
        p = (int(start_cell[0]), int(start_cell[1]))
        d = int(direction)

        for _ in range(max_len - 1):
            trans = self._rail_get_transitions(p, d)
            n_trans = fast_count_nonzero(trans)
            if n_trans != 1:
                break  # decision point or dead end

            d = int(fast_argmax(trans))
            p = get_new_position(p, d)

            if (p[0] < 0 or p[0] >= self.env.height
                    or p[1] < 0 or p[1] >= self.env.width):
                break

            cells.append((int(p[0]), int(p[1])))

        return cells

        # ─── TREE EDGE ENRICHMENT (override) ─────────────────────────────────
    def _build_corridor_edge_payload(
        self,
        handle,
        src_idx,
        dst_idx,
        src_pos,
        src_dir,
        src_depth,
        dst_pos,
        dst_dir,
        corridor,
        distance_map,
    ):
        """Override: add conflict-aware fields to each tree edge.
        
        Each edge in the local search tree is enriched with:
          - conflict_density:  fraction of corridor cells with predicted conflicts
          - priority_yield:    how many corridor agents have lower priority than me
          - sp_dominance:      do I have priority advantage on this edge?
        
        These fields are consumed by the tree encoder (no new architecture
        needed; the encoder already attends over edge fields).
        """
        # 1. Get parent's edge payload (already rich with merge/deadlock context)
        edge = super()._build_corridor_edge_payload(
            handle=handle,
            src_idx=src_idx,
            dst_idx=dst_idx,
            src_pos=src_pos,
            src_dir=src_dir,
            src_depth=src_depth,
            dst_pos=dst_pos,
            dst_dir=dst_dir,
            corridor=corridor,
            distance_map=distance_map,
        )

        # 2. Compute new conflict-aware fields
        edge_cells = edge.get("cells", [])
        edge_agents = edge.get("agents", [])

        # ── conflict_density ──────────────────────────────────────────────
        # Fraction of edge cells that appear in predicted vertex reservations
        # of OTHER agents (any time within horizon).
        if self._conflict_predictor is not None and edge_cells:
            n_conflict_cells = 0
            for cell in edge_cells:
                cell_t = (int(cell[0]), int(cell[1]))
                # Check if this cell is reserved by anyone else within horizon
                conflict_found = False
                for t in range(1, self._conflict_horizon + 1):
                    users = self._conflict_predictor._vertex_res.get(
                        (cell_t[0], cell_t[1], t), []
                    )
                    others = [h for h in users if h != handle]
                    if others:
                        conflict_found = True
                        break
                if conflict_found:
                    n_conflict_cells += 1
            edge["conflict_density"] = float(n_conflict_cells / max(1, len(edge_cells)))
        else:
            edge["conflict_density"] = 0.0

        # ── priority_yield ────────────────────────────────────────────────
        # How many edge_agents have LOWER priority than me?
        # → If high, this means others should yield to me on this edge.
        if self._conflict_predictor is not None and edge_agents:
            my_d = self._conflict_predictor._sp_distance_at_start.get(
                handle, float("inf")
            )
            n_lower_priority = 0
            for other_handle in edge_agents:
                other_d = self._conflict_predictor._sp_distance_at_start.get(
                    other_handle, float("inf")
                )
                # Lower priority = larger remaining distance
                if np.isfinite(other_d) and np.isfinite(my_d) and other_d > my_d:
                    n_lower_priority += 1
            edge["priority_yield"] = float(
                n_lower_priority / max(1, len(edge_agents))
            )
        else:
            edge["priority_yield"] = 0.0

        # ── sp_dominance ──────────────────────────────────────────────────
        # Do I have a clear distance advantage on this edge?
        # Computed as: (1.0 - dst_dist / max_other_dist) for agents on this edge.
        # Range [0, 1]: 1.0 = I'm clearly closest to goal among edge users.
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

        
