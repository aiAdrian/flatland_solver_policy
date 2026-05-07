"""
HierarchicalRoutesObservation
=============================

Extension of `DecisionPointObservation` that adds a compact, sparse view of the
top-K most relevant other agents WITHOUT exposing handles / ids / positions.
Designed to feed the Decider policy with `Specialist` sub-modules
(see HIERARCHICAL_DECIDER_ARCHITECTURE.md, sections 4-5).

Output per `get(handle)`:
    (feature_vector_72D, opp_handles_list)

Layout of the 72D vector:

    [ 0 - 47]  Base 48D from DecisionPointObservation (unchanged):
               - decision_type, shortest-path-hint, 3 branch blocks (route info!),
                 2 merge blocks, state/action one-hots, local deadlock,
                 coordination soft-signals.
    [48 - 71]  Sparse neighbor block: K=4 classified neighbors x 6D each.

Per-neighbor 6D layout:
    [exists, class_oncoming, class_merging, class_local, distance_norm, ttc_norm]

Notes
-----
* Routes are already represented in base[4-21] (3 branches x 6D). We do NOT
  duplicate them here - Flatland has at most 3 outgoing transitions at a
  decision point. Specialists read base[4-21] directly.
* The list `opp_handles_list` is the SAME K=4 ordering (after relevance sort).
  The Decider policy uses it to look up the K=4 most relevant agents'
  `global_metric` vectors from its CommBuffer.
* No agent_id / handle / position is encoded numerically - only the 3-bit
  class + scalar distances + ttc.
"""

from typing import List, Tuple

import numpy as np

from flatland.envs.fast_methods import fast_count_nonzero, fast_argmax
from flatland.core.grid.grid4_utils import get_new_position

from marl_attention_temporal_observation.decision_point_observation import (
    DecisionPointObservation,
)


# Per-neighbor block size and ordering.
NEIGHBOR_BLOCK_SIZE = 6
NEIGHBOR_K = 4
NEIGHBOR_TOTAL = NEIGHBOR_BLOCK_SIZE * NEIGHBOR_K

# Class indices inside the per-neighbor block.
NEIGHBOR_IDX_EXISTS = 0
NEIGHBOR_IDX_ONCOMING = 1
NEIGHBOR_IDX_MERGING = 2
NEIGHBOR_IDX_LOCAL = 3
NEIGHBOR_IDX_DISTANCE_NORM = 4
NEIGHBOR_IDX_TTC_NORM = 5


class HierarchicalRoutesObservation(DecisionPointObservation):
    """Decision-point observation + sparse top-K classified neighbors block."""

    OBS_SIZE = DecisionPointObservation.OBS_SIZE + NEIGHBOR_TOTAL  # 72

    # Neighbor-classification reach radius (in grid cells, manhattan).
    LOCAL_RADIUS = 6
    # Conflict look-ahead horizon for ONCOMING / MERGING detection.
    CONFLICT_HORIZON = 6

    def __init__(self, top_k: int = NEIGHBOR_K, local_radius: int = 6):
        super().__init__()
        self.feature_len = HierarchicalRoutesObservation.OBS_SIZE
        self.top_k = int(top_k)
        self.LOCAL_RADIUS = int(local_radius)
        if not getattr(type(self), "_banner_printed", False):
            print(">> HierarchicalRoutesObservation loaded.")
            type(self)._banner_printed = True

    @staticmethod
    def getObservationSize() -> int:
        return HierarchicalRoutesObservation.OBS_SIZE

    # ------------------------------------------------------------------
    # Main hook
    # ------------------------------------------------------------------
    def get(self, handle: int = 0) -> Tuple[np.ndarray, List[int]]:
        # 1) Base features from parent. The parent uses `self.feature_len` for
        # allocation, which we overrode to 72 -> parent already returns a 72D
        # vector with [48..71] zero. We use it as our buffer.
        base_features, opp_handles = super().get(handle)
        feat = np.zeros(self.feature_len, dtype=np.float32)
        cap = min(int(np.asarray(base_features).shape[0]), self.feature_len)
        feat[:cap] = np.asarray(base_features, dtype=np.float32)[:cap]
        # Make absolutely sure neighbor block is zeroed before we fill it.
        feat[DecisionPointObservation.OBS_SIZE :] = 0.0

        agent = self.env.agents[handle]
        pos = agent.position if agent.position is not None else agent.initial_position
        direction = agent.direction if agent.direction is not None else agent.initial_direction
        if pos is None or direction is None:
            return feat, []

        # 2) Build candidate neighbor list from the env (NOT just opp_handles
        #    from parent - parent only exposes "DFS-seen" agents which is too
        #    restrictive). We use a simple manhattan radius, then classify.
        candidates = self._collect_candidate_neighbors(handle, pos)

        # 3) Classify + score each candidate.
        scored: List[Tuple[float, int, np.ndarray]] = []
        for other_handle in candidates:
            block, score = self._classify_neighbor(handle, pos, direction, other_handle)
            if block is None:
                continue
            scored.append((score, other_handle, block))

        # 4) Top-K by relevance score.
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[: self.top_k]

        # 5) Write blocks to feature vector (zero-pad rest).
        for slot, (_score, _h, block) in enumerate(scored):
            base = DecisionPointObservation.OBS_SIZE + slot * NEIGHBOR_BLOCK_SIZE
            feat[base : base + NEIGHBOR_BLOCK_SIZE] = block

        # 6) Return ordered handle list (same order as the K blocks). Used by
        #    the Decider policy to read `global_metric` from its CommBuffer.
        ordered_handles = [h for (_s, h, _b) in scored]
        return feat, ordered_handles

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _collect_candidate_neighbors(self, handle: int, pos) -> List[int]:
        """All other agents within manhattan LOCAL_RADIUS of `pos`."""
        out: List[int] = []
        for other in self.env.agents:
            if other.handle == handle:
                continue
            opos = other.position if other.position is not None else other.initial_position
            if opos is None:
                continue
            if abs(opos[0] - pos[0]) + abs(opos[1] - pos[1]) > self.LOCAL_RADIUS:
                continue
            out.append(other.handle)
        return out

    def _classify_neighbor(self, handle: int, my_pos, my_dir, other_handle: int):
        """Return (block_6D, relevance_score) for one neighbor or (None, 0)."""
        other = self.env.agents[other_handle]
        opos = other.position if other.position is not None else other.initial_position
        odir = other.direction if other.direction is not None else other.initial_direction
        if opos is None or odir is None:
            return None, 0.0

        dist = abs(opos[0] - my_pos[0]) + abs(opos[1] - my_pos[1])
        # Distance normalisation: clamp to LOCAL_RADIUS.
        dist_norm = float(np.clip(dist / max(1, self.LOCAL_RADIUS), 0.0, 1.0))

        # ONCOMING: opposing direction and reachable on my forward rollout.
        is_oncoming = self._is_oncoming(my_pos, my_dir, opos, odir)
        # MERGING: not opposing, but same/parallel direction with a track that
        # merges into mine within horizon.
        is_merging = (not is_oncoming) and self._is_merging(my_pos, my_dir, opos, odir)
        # LOCAL: nearby but no direct conflict.
        is_local = (not is_oncoming) and (not is_merging)

        # TTC: rough time-to-collision proxy = remaining cells until our
        # forward rollout meets `opos`.
        ttc_steps = self._estimate_ttc(my_pos, my_dir, opos)
        if ttc_steps < 0 or ttc_steps > self.CONFLICT_HORIZON:
            ttc_norm = 0.0
        else:
            ttc_norm = float(np.clip(1.0 - (ttc_steps / float(self.CONFLICT_HORIZON)), 0.0, 1.0))

        block = np.zeros(NEIGHBOR_BLOCK_SIZE, dtype=np.float32)
        block[NEIGHBOR_IDX_EXISTS] = 1.0
        block[NEIGHBOR_IDX_ONCOMING] = 1.0 if is_oncoming else 0.0
        block[NEIGHBOR_IDX_MERGING] = 1.0 if is_merging else 0.0
        block[NEIGHBOR_IDX_LOCAL] = 1.0 if is_local else 0.0
        block[NEIGHBOR_IDX_DISTANCE_NORM] = dist_norm
        block[NEIGHBOR_IDX_TTC_NORM] = ttc_norm

        # Relevance score: higher = more important. Oncoming at low TTC top.
        score = (
            2.0 * float(is_oncoming) * (0.5 + ttc_norm)
            + 1.2 * float(is_merging) * (0.5 + ttc_norm)
            + 0.4 * float(is_local) * (1.0 - dist_norm)
            + 0.2 * ttc_norm
        )
        return block, float(score)

    @staticmethod
    def _is_opposite_direction(d1: int, d2: int) -> bool:
        return (d1 + 2) % 4 == d2

    def _is_oncoming(self, my_pos, my_dir, opos, odir) -> bool:
        """Other agent moves towards me on my forward path within horizon."""
        if not self._is_opposite_direction(my_dir, odir):
            return False
        cur_pos = my_pos
        cur_dir = my_dir
        for _ in range(self.CONFLICT_HORIZON):
            trans = self.env.rail.get_transitions(*cur_pos, cur_dir)
            n = fast_count_nonzero(trans)
            if n == 0:
                return False
            if n > 1:
                # branch -> stop rollout (uncertain)
                return False
            cur_dir = fast_argmax(trans)
            cur_pos = get_new_position(cur_pos, cur_dir)
            if cur_pos == opos:
                return True
        return False

    def _is_merging(self, my_pos, my_dir, opos, odir) -> bool:
        """Other agent's forward rollout joins my track within horizon."""
        cur_pos = opos
        cur_dir = odir
        my_track = set()
        # Build my forward track for HORIZON steps.
        cp = my_pos
        cd = my_dir
        for _ in range(self.CONFLICT_HORIZON):
            my_track.add(cp)
            tr = self.env.rail.get_transitions(*cp, cd)
            if fast_count_nonzero(tr) != 1:
                break
            cd = fast_argmax(tr)
            cp = get_new_position(cp, cd)
        # Walk other's forward path; merge if it enters my_track at a cell
        # other than its current position.
        for step in range(1, self.CONFLICT_HORIZON + 1):
            tr = self.env.rail.get_transitions(*cur_pos, cur_dir)
            if fast_count_nonzero(tr) != 1:
                return False
            cur_dir = fast_argmax(tr)
            cur_pos = get_new_position(cur_pos, cur_dir)
            if cur_pos in my_track:
                return True
        return False

    def _estimate_ttc(self, my_pos, my_dir, opos) -> int:
        """Cells until my forward rollout reaches `opos`. -1 if unreachable."""
        cur_pos = my_pos
        cur_dir = my_dir
        for step in range(1, self.CONFLICT_HORIZON + 1):
            tr = self.env.rail.get_transitions(*cur_pos, cur_dir)
            if fast_count_nonzero(tr) == 0:
                return -1
            if fast_count_nonzero(tr) > 1:
                return -1
            cur_dir = fast_argmax(tr)
            cur_pos = get_new_position(cur_pos, cur_dir)
            if cur_pos == opos:
                return step
        return -1

    # ------------------------------------------------------------------
    # get_many: keep parent's batch logic (sets agent_map / max_dist) and
    # then dispatch our own per-handle get().
    # ------------------------------------------------------------------
    def get_many(self, handles: list = None):
        if handles is None:
            handles = list(range(len(self.env.agents)))

        self.agent_map = np.zeros((self.env.height, self.env.width), dtype=np.int32) - 1
        for agent in self.env.agents:
            if agent.position is not None:
                self.agent_map[agent.position] = agent.handle

        distance_map = self.env.distance_map.get()
        finite = distance_map[np.isfinite(distance_map)]
        if finite.size > 0:
            self._max_dist = max(float(np.max(finite)), 1.0)
        else:
            self._max_dist = 1.0

        for agent in self.env.agents:
            if not hasattr(agent, "opp_agent_handles"):
                agent.opp_agent_handles = []
            if not hasattr(agent, "cur_opp_agent_handles"):
                agent.cur_opp_agent_handles = []

        result = []
        for handle in handles:
            obs_self, opp_handles = self.get(handle)
            result.append((obs_self, opp_handles))

        for agent in self.env.agents:
            agent.opp_agent_handles = agent.cur_opp_agent_handles

        return result
