"""Decision-point observation used by MAPPO.

Layout:
- BASE_OBS_SIZE=15: local agent state features
- TREE PAYLOAD: Decision-Point Graph with three node types:
  - INIT (type=0): Agent initialization/spawn
  - SWITCH (type=1): Route choice (num_transitions > 1)
  - PRE_M (type=2): Pre-merge decision (forward vs wait)
  
  Corridors are compressed into edges. Merges modeled as edge context (merge_conflict flag).
  Depth counts decision-point transitions, not cell hops.
  
  Each node contains deadlock_risk, deadlock_ahead, deadlock_hard_block signals.
  Each edge contains merge_conflict flag and merge_incoming_degree for merge context.
  
- exported via env.dev_tree_dict[handle]
"""

# pyright: reportMissingImports=false

import os
import time
from enum import IntEnum

import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_argmax, fast_count_nonzero
from flatland.envs.step_utils.states import TrainState

from marl_attention_temporal_observation.decision_point_utils import DecisionPointUtils

_UNREACHABLE = float("inf") 


class NodeType(IntEnum):
    """Decision-point node types for local-search-tree.
    
    Nodes exist ONLY at real decision points:
    - INIT: Agent spawn/entry (virtual decision)
    - SWITCH: Multiple route choices (num_transitions > 1, not pre-merge)
    - PRE_M: Pre-merge decision (agent chooses forward vs wait)
    
    MERGE is NOT a node—it's modeled as edge context (no agent decision possible).
    Corridors are compressed into edges between decision nodes.
    """
    INIT = 0      # Initialization / agent spawn
    SWITCH = 1    # Switch/route choice (num_transitions > 1)
    PRE_M = 2     # Pre-merge decision (one exit, next node has multiple entries)


class DecisionPointObservation(ObservationBuilder):
    _get_many_call_count = 0
    _last_100_features = []  # List of np.arrays (n_agents, n_features)
    _last_100_tree_stats = []  # List of tree_stats pro Episode
    _last_obs_fn_perf_report = None

    # Export 15 base features.
    # Legacy lifecycle duplicates were removed; deadlock signals live in tree payload.
    BASE_OBS_SIZE = 15
    OBS_SIZE = BASE_OBS_SIZE
    # Legacy alias; active runtime cap is configured via self.local_search_max_nodes.
    MAX_NODES = 48

    FEATURE_GROUPS_DOC = [
        ("[0-2]",   "path_left/forward/right",  "1 if relative transition exists"),
        ("[3-5]",   "delta_left/forward/right", "exp-squashed gap-to-best successor in [-1,1] (0=best, <0=worse)"),
        ("[6-8]",   "st_3/st_4/st_6",           "TrainState READY_TO_DEPART + MALFUNCTION + not-started"),
        ("[9]",     "priority_rank",            "normalized rank by remaining distance"),
        ("[10-11]", "merge/switch",             "cell semantics (redundant lifecycle flags removed)"),
        ("[12-14]", "sp_left/sp_forward/sp_right", "shortest-path action hint one-hot"),
        ("payload", "raw_tree_payload",         "exported separately via env.dev_tree_dict[handle]. Nodes/edges include deadlock_risk."),
    ]

    # Canonical base-feature specification for indices 0..14 (15D base obs).
    # Removed [12]is_started (inverse of [8]st_6) and [13]is_done (duplicate of [7]st_4).
    # Deadlock features moved to tree payload (node and edge features).
    # Keep this list in sync with get() and runtime summary names.
    BASE_FEATURE_SPECS = [
        (0,  "path_left",            "1 if relative left transition exists else 0"),
        (1,  "path_forward",         "1 if relative forward transition exists else 0"),
        (2,  "path_right",           "1 if relative right transition exists else 0"),
        (3,  "delta_left",           "exp-squashed (best_successor_dist - left_dist) in [-1,1], else -1 if no transition"),
        (4,  "delta_forward",        "exp-squashed (best_successor_dist - forward_dist) in [-1,1], else -1 if no transition"),
        (5,  "delta_right",          "exp-squashed (best_successor_dist - right_dist) in [-1,1], else -1 if no transition"),
        (6,  "st_3",                 "TrainState READY_TO_DEPART one-hot (state_value==3)"),
        (7,  "st_4",                 "TrainState MALFUNCTION one-hot (only alive state)"),
        (8,  "st_6",                 "1 if agent is not started (position is None) else 0"),
        (9,  "priority_rank",        "normalized distance-rank priority"),
        (10, "is_pre_merge",         "1 if one step before merge-conflict point else 0"),
        (11, "is_switch",            "1 if current cell has >1 transitions else 0"),
        (12, "sp_left",              "shortest-path hint one-hot: left"),
        (13, "sp_forward",           "shortest-path hint one-hot: forward"),
        (14, "sp_right",             "shortest-path hint one-hot: right"),
    ]

    def __init__(self,
                 debug: bool = False,
                 search_depth: int = 5,
                 observation_profile: str = "local_tree_encoder",
                 use_trainable_tree_encoder: bool = True):
        super().__init__()
        if debug:
            os.environ["DEBUG_OBSERVATION"] = "1"
        # Core observation configuration used throughout get()/local search.
        self.search_depth = max(1, int(search_depth))
        self.observation_profile = observation_profile
        self.use_trainable_tree_encoder = bool(use_trainable_tree_encoder)
        self.local_search_min_search_depth = 8
        # Local-tree search control to avoid branch explosion at higher depths.
        # Up to depth 1: expand all transitions.
        # From depth >= 2: always keep shortest-path branch and sample side branches.
        self.local_search_random_start_depth = 2
        self.local_search_max_side_branches = 3
        self.local_search_distance_bias = 2.0
        # Optional advanced controls for deeper searches.
        self.local_search_mode = "stochastic"  # stochastic | mcts
        self.local_search_mcts_rollouts = 6
        self.local_search_mcts_horizon = 4
        self.local_search_ucb_c = 1.2
        self.local_search_contract_depth = 7
        # Enable corridor contraction by default so local search reaches
        # downstream decision points within limited node budgets.
        self.local_search_disable_corridor_contraction = False
        self.local_search_max_nodes = 72
        self.local_search_min_nodes = 24
        self.local_search_adaptive_budget = True
        self.local_search_adaptive_branch_bonus = 6
        self.local_search_adaptive_conflict_bonus = 8
        self.local_search_adaptive_depth_bonus = 2
        self.local_search_deadlock_probe_depth = 14
        self.local_search_deadlock_max_states = 256
        self.local_tree_clip_features = True
        # Debug-only render overlay. Handle 0 exports pseudo-agent cell sets:
        # 0=all node cells, 1=pre-merge, 2=switch, 3/4=even/odd corridor cells.
        self.debug_tree_overlay_enabled = True
        # Lightweight function profiler for observation hot paths.
        self.obs_func_profile_enabled = str(os.getenv('FLATLAND_OBS_FUNC_PROFILE', '1')).strip().lower() in ('1', 'true', 'yes', 'on')
        self.obs_func_profile_sample_every = max(1, int(os.getenv('FLATLAND_OBS_FUNC_PROFILE_SAMPLE_EVERY', '8')))
        self.obs_func_profile_interval = max(1, int(os.getenv('FLATLAND_OBS_FUNC_PROFILE_INTERVAL_EPISODES', '20')))
        self._obs_func_profile_call_idx = 0
        self._obs_profile_active = False
        self._obs_func_prof = {
            'get': {'sum': 0.0, 'count': 0},
            'get_many': {'sum': 0.0, 'count': 0},
            'local_search': {'sum': 0.0, 'count': 0},
            'deadlock_profile': {'sum': 0.0, 'count': 0},
            'base_features': {'sum': 0.0, 'count': 0},
            'debug_overlay': {'sum': 0.0, 'count': 0},
        }
        self.env = None
        self.agent_map = None
        self._print_feature_layout_doc()

    def _obs_prof_add(self, key: str, dt: float):
        if not self.obs_func_profile_enabled:
            return
        bucket = self._obs_func_prof.get(key)
        if bucket is None:
            return
        bucket['sum'] += float(dt)
        bucket['count'] += 1
 
    def set_env(self, env):
        super().set_env(env)
        self.env = env

    def reset(self):
        self.agent_map = np.zeros((self.env.height, self.env.width), dtype=np.int32) - 1

    @staticmethod
    def _dir_to_rel_bin(current_dir: int, next_dir: int) -> int:
        """Map absolute next direction to relative bin: left=0, forward=1, right=2, other=3."""
        if current_dir is None or next_dir is None:
            return 3
        delta = (int(next_dir) - int(current_dir)) % 4
        if delta == 3:
            return 0
        if delta == 0:
            return 1
        if delta == 1:
            return 2
        return 3

    @classmethod
    def _rel_dir_one_hot(cls, current_dir: int, next_dir: int) -> tuple:
        rel_bin = cls._dir_to_rel_bin(current_dir, next_dir)
        return (
            1.0 if rel_bin == 0 else 0.0,
            1.0 if rel_bin == 1 else 0.0,
            1.0 if rel_bin == 2 else 0.0,
        )

    @staticmethod
    def _relative_dir_order(current_dir: int) -> tuple:
        if current_dir is None:
            return 0, 1, 2, 3
        current_dir = int(current_dir)
        return (
            (current_dir - 1) % 4,  # left
            current_dir,            # forward
            (current_dir + 1) % 4,  # right
            (current_dir + 2) % 4,  # backward / other
        )

    @classmethod
    def _sort_branch_candidates_relative(cls, candidates: list, current_dir: int) -> list:
        order_index = {d: i for i, d in enumerate(cls._relative_dir_order(current_dir))}
        return sorted(
            candidates,
            key=lambda c: (order_index.get(int(c[0]), 99), float(c[2]) if len(c) > 2 else 0.0),
        )

    def _safe_distance(self, handle, position, direction, distance_map, default=np.inf):
        if distance_map is None:
            return default
        if position is None or direction is None:
            return default
        return float(distance_map[handle, position[0], position[1], direction])

    @staticmethod
    def _distance_to_unit(distance: float, max_dist: float) -> float:
        if not np.isfinite(distance):
            return 1.0
        denom = max(1.0, float(max_dist))
        return float(np.clip(float(distance) / denom, 0.0, 1.0))

    @staticmethod
    def _progress_delta_to_unit(root_dist: float, dst_dist: float, max_dist: float) -> float:
        if not np.isfinite(root_dist) and np.isfinite(dst_dist):
            return 1.0
        if not np.isfinite(root_dist) or not np.isfinite(dst_dist):
            return 0.0
        denom = max(1.0, float(max_dist))
        return float(np.clip((float(root_dist) - float(dst_dist)) / denom, -1.0, 1.0))

    @staticmethod
    def _exp_squash_signed(value: float, scale: float = 8.0) -> float:
        """Map unbounded signed values smoothly to (-1, 1) while preserving magnitude order."""
        if not np.isfinite(value):
            return 0.0
        s = max(1e-6, float(scale))
        x = float(value)
        if x == 0.0:
            return 0.0
        return float(np.sign(x) * (1.0 - np.exp(-abs(x) / s)))

    @staticmethod
    def _pos_tuple(pos):
        if pos is None:
            return None
        return (int(pos[0]), int(pos[1]))

    def _rail_get_transitions(self, pos, direction):
        """Compatibility wrapper for Flatland transition APIs.

        Standard signature: get_transitions(row, col, direction) → tuple(4-8)
        """
        p = self._pos_tuple(pos)
        d = int(direction)
        return self.env.rail.get_transitions(p[0], p[1], d)

    def _agent_at_pos(self, pos) -> int:
        if self.agent_map is None:
            return -1
        p = self._pos_tuple(pos)
        return int(self.agent_map[p[0], p[1]])

    def _mcts_rollout_score(self, handle, start_pos, start_dir, start_depth, horizon, distance_map):
        """Small Monte-Carlo rollout score for one root branch.

        Uses a light UCT-style policy over local successor choices to keep
        selection robust while staying compute-bounded.
        """
        if self.env is None or self.env.rail is None:
            return -1e9

        pos = start_pos
        direction = int(start_dir)
        score = 0.0
        max_steps = max(1, int(horizon))

        for _ in range(max_steps):
            dist = self._safe_distance(handle, pos, direction, distance_map)
            if np.isfinite(dist):
                score += 1.0 / (1.0 + dist)
            else:
                score -= 0.05

            if self.agent_map is not None:
                other_idx = self._agent_at_pos(pos)
                if other_idx != -1 and other_idx != handle:
                    score -= 0.7
                    other_dir = self.env.agents[other_idx].direction
                    if other_dir is not None and DecisionPointUtils.is_opposite_direction(direction, other_dir):
                        score -= 0.8

            transitions = self._rail_get_transitions(pos, direction)

            choices = [nd for nd in range(4) if transitions[nd]]
            if not choices:
                score -= 1.0
                break

            # Soft distance-biased stochastic rollout policy.
            cand = []
            for nd in choices:
                np_pos = get_new_position(pos, nd)
                ndist = self._safe_distance(handle, np_pos, nd, distance_map)
                cand.append((nd, np_pos, ndist))
            dvals = np.array([c[2] if np.isfinite(c[2]) else 1000.0 for c in cand], dtype=np.float64)
            dmin = float(np.min(dvals))
            closeness = 1.0 / (1.0 + np.maximum(0.0, dvals - dmin))
            alpha = max(0.1, float(self.local_search_distance_bias))
            weights = np.power(closeness, alpha)
            wsum = float(np.sum(weights))
            if wsum <= 0.0 or not np.isfinite(wsum):
                probs = np.full(len(cand), 1.0 / len(cand), dtype=np.float64)
            else:
                probs = weights / wsum

            idx = int(np.random.choice(len(cand), p=probs))
            direction, pos, _ = cand[idx]

        return score

    def _compute_adaptive_node_budget(
        self,
        handle,
        start_pos,
        start_dir,
        depth_limit,
        transition_cache=None,
        incoming_degree_cache=None,
    ) -> int:
        max_nodes = int(getattr(self, "local_search_max_nodes", 48))
        min_nodes = int(getattr(self, "local_search_min_nodes", 24))
        if not bool(getattr(self, "local_search_adaptive_budget", True)):
            return max(min_nodes, max_nodes)

        bonus = int(max(0, int(depth_limit) - 3)) * int(getattr(self, "local_search_adaptive_depth_bonus", 2))
        transitions = self._rail_get_transitions(start_pos, start_dir)
        if fast_count_nonzero(transitions) > 1:
            bonus += int(getattr(self, "local_search_adaptive_branch_bonus", 6))

        budget = max_nodes + bonus
        return max(min_nodes, min(max_nodes + 2 * int(getattr(self, "local_search_adaptive_depth_bonus", 2)), budget))

    def _select_local_search_branches(
        self,
        handle,
        depth,
        current_pos,
        current_dir,
        transitions,
        distance_map,
    ) -> list:
        candidates = []
        for nd in range(4):
            if not transitions[nd]:
                continue
            np_pos = get_new_position(current_pos, nd)
            ndist = self._safe_distance(handle, np_pos, nd, distance_map)
            candidates.append((nd, np_pos, ndist))

        if not candidates:
            return []
        if len(candidates) == 1:
            return candidates

        ordered = self._sort_branch_candidates_relative(candidates, current_dir)
        shortest = min(ordered, key=lambda c: (c[2] if np.isfinite(c[2]) else float("inf")))
        side = [c for c in ordered if c is not shortest]

        if int(depth) < int(getattr(self, "local_search_random_start_depth", 2)):
            return ordered

        k_side = max(0, int(getattr(self, "local_search_max_side_branches", 1)))
        if k_side == 0 or not side:
            return [shortest]

        side_sorted = sorted(side, key=lambda c: (c[2] if np.isfinite(c[2]) else float("inf")))
        return [shortest] + side_sorted[:k_side]

    def _contract_corridor_segment(self, handle, pos, direction, depth, depth_limit, target):
        cur_pos = pos
        cur_dir = int(direction)
        edge_len = 1
        target_on_edge = bool(target is not None and cur_pos == target)
        if bool(getattr(self, "local_search_disable_corridor_contraction", False)):
            return cur_pos, cur_dir, edge_len, target_on_edge
        visited = set()

        while int(depth) + edge_len < int(depth_limit):
            if target_on_edge:
                break
            state = (int(cur_pos[0]), int(cur_pos[1]), int(cur_dir))
            if state in visited:
                break
            visited.add(state)

            transitions = self._rail_get_transitions(cur_pos, cur_dir)
            if fast_count_nonzero(transitions) != 1:
                break

            ndir = int(fast_argmax(transitions))
            next_pos = get_new_position(cur_pos, ndir)
            if next_pos[0] < 0 or next_pos[0] >= self.env.height or next_pos[1] < 0 or next_pos[1] >= self.env.width:
                break

            cur_pos = next_pos
            cur_dir = ndir
            edge_len += 1
            target_on_edge = bool(target is not None and cur_pos == target)

        return cur_pos, cur_dir, edge_len, target_on_edge

    def _local_node_type(self, pos, direction, is_root=False):
        transitions = self._rail_get_transitions(pos, direction)
        n_trans = fast_count_nonzero(transitions)
        if is_root:
            return 0  # START
        if n_trans > 1:
            return 1  # SWITCH
        if n_trans != 1:
            return None

        ndir = int(fast_argmax(transitions))
        next_pos = get_new_position(pos, ndir)
        if next_pos[0] < 0 or next_pos[0] >= self.env.height or next_pos[1] < 0 or next_pos[1] >= self.env.width:
            return None

        next_trans = self._rail_get_transitions(next_pos, ndir)
        if fast_count_nonzero(next_trans) != 1:
            return None

        for d_other in range(4):
            if d_other == ndir:
                continue
            if fast_count_nonzero(self._rail_get_transitions(next_pos, d_other)) > 1:
                return 2  # PRE_MERGE
        return None

    def _build_local_node_payload(self, handle, ntype, npos, ndir, depth_value, max_depth):
        transitions_local = self._rail_get_transitions(npos, ndir)
        num_transitions_local = int(fast_count_nonzero(transitions_local))
        incoming_degree_local = int(self._incoming_degree(npos))

        incoming_agent_count = 0
        has_oncoming = False
        occ_handle = self._agent_at_pos(npos)
        if occ_handle != -1 and occ_handle != handle:
            incoming_agent_count = 1
            other_dir = self.env.agents[occ_handle].direction
            if other_dir is not None and DecisionPointUtils.is_opposite_direction(ndir, other_dir):
                has_oncoming = True

        deadlock_profile = self._calculate_deadlock_profile(
            handle=handle,
            pos=npos,
            direction=ndir,
            max_depth=min(12, max(4, int(max_depth))),
        )

        node_type_value = int(ntype)
        is_start = bool(node_type_value == 0)
        is_switch = bool(node_type_value == 1)
        is_pre_merge = bool(node_type_value == 2)
        node_type_name = "start" if is_start else ("switch" if is_switch else "pre_merge")

        return {
            "type": int(ntype),
            "node_type_name": node_type_name,
            "position": (int(npos[0]), int(npos[1])),
            "cells": [(int(npos[0]), int(npos[1]))],
            "direction": int(ndir),
            "is_start": bool(is_start),
            "is_switch": bool(is_switch),
            "is_pre_merge": bool(is_pre_merge),
            "depth": int(depth_value),
            "num_transitions": int(num_transitions_local),
            "is_branch": bool(is_switch or num_transitions_local > 1),
            "has_oncoming": bool(has_oncoming),
            "incoming_agent_count": int(incoming_agent_count),
            "has_agents_encountered": bool(incoming_agent_count > 0),
            "backward_inflow_count": int(max(0, incoming_degree_local - 1)),
            "deadlock_risk": float(deadlock_profile.get("risk", 0.0)),
            "deadlock_distance_norm": float(deadlock_profile.get("deadlock_distance_norm", 0.0)),
            "deadlock_hard_distance_norm": float(deadlock_profile.get("hard_block_distance_norm", 0.0)),
            "deadlock_exists_within_probe": bool(int(deadlock_profile.get("min_deadlock_depth", -1)) >= 0),
            "alternative_routes_count": int(max(1 if is_pre_merge else 0, max(0, num_transitions_local - 1))),
        }

    def _walk_corridor_until_decision(
        self,
        handle,
        src_pos,
        src_dir,
        first_dir,
        agent_target,
        distance_map,
    ):
        edge_path = []
        edge_agents = set()
        same_dir_handles = set()
        oncoming_handles = set()
        forward_handles = set()
        backward_handles = set()
        edge_len = 0
        min_dist = float("inf")
        target_on_edge = False
        p, d = tuple(src_pos), int(src_dir)
        step_dir = int(first_dir)
        max_corridor_steps = max(256, 2 * (self.env.height + self.env.width))
        visited_states = set()

        while edge_len < max_corridor_steps:
            state = (tuple(p), int(d), int(step_dir))
            if state in visited_states:
                break
            visited_states.add(state)

            transitions = self._rail_get_transitions(p, d)
            if not transitions[step_dir]:
                break

            np_pos = get_new_position(p, step_dir)
            if np_pos[0] < 0 or np_pos[0] >= self.env.height or np_pos[1] < 0 or np_pos[1] >= self.env.width:
                break

            edge_path.append((np_pos, step_dir))

            if self.agent_map is not None:
                aidx = int(self.agent_map[np_pos[0], np_pos[1]])
                if aidx != -1 and aidx != handle:
                    edge_agents.add(aidx)
                    other_agent = self.env.agents[aidx]
                    if other_agent.direction == step_dir:
                        same_dir_handles.add(aidx)
                        forward_handles.add(aidx)
                    else:
                        oncoming_handles.add(aidx)
                        backward_handles.add(aidx)

            if agent_target is not None and tuple(np_pos) == tuple(agent_target):
                target_on_edge = True

            if distance_map is not None:
                dist = distance_map[handle, np_pos[0], np_pos[1], step_dir]
                if np.isfinite(dist):
                    min_dist = min(min_dist, dist)

            edge_len += 1
            p, d = np_pos, step_dir

            dst_type = self._local_node_type(p, d)
            if dst_type is not None:
                return {
                    "complete": True,
                    "dst_pos": tuple(p),
                    "dst_dir": int(d),
                    "dst_type": int(dst_type),
                    "edge_path": edge_path,
                    "edge_len": int(edge_len),
                    "edge_agents": sorted(edge_agents),
                    "same_dir_handles": sorted(same_dir_handles),
                    "oncoming_handles": sorted(oncoming_handles),
                    "forward_handles": sorted(forward_handles),
                    "backward_handles": sorted(backward_handles),
                    "min_dist_to_target": min_dist if min_dist != float("inf") else None,
                    "target_on_edge": bool(target_on_edge),
                }

            transitions_next = self._rail_get_transitions(p, d)
            next_dirs = [ndir for ndir in range(4) if transitions_next[ndir]]
            if len(next_dirs) != 1:
                break
            step_dir = int(next_dirs[0])

        return {
            "complete": False,
            "edge_path": edge_path,
            "edge_len": int(edge_len),
            "edge_agents": sorted(edge_agents),
            "same_dir_handles": sorted(same_dir_handles),
            "oncoming_handles": sorted(oncoming_handles),
            "forward_handles": sorted(forward_handles),
            "backward_handles": sorted(backward_handles),
            "min_dist_to_target": min_dist if min_dist != float("inf") else None,
            "target_on_edge": bool(target_on_edge),
        }

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
        action_feature = None
        edge_path = corridor.get("edge_path", [])
        if edge_path:
            _first_pos, first_dir = edge_path[0]
            rel_dir = (int(first_dir) - int(src_dir)) % 4
            if rel_dir == 1:
                action_feature = 1
            elif rel_dir == 0:
                action_feature = 0
            elif rel_dir == 3:
                action_feature = -1

        rel_dir_bin = 1
        action_left = 0.0
        action_forward = 1.0
        action_right = 0.0
        if action_feature == -1:
            rel_dir_bin = 0
            action_left, action_forward, action_right = 1.0, 0.0, 0.0
        elif action_feature == 0:
            rel_dir_bin = 1
            action_left, action_forward, action_right = 0.0, 1.0, 0.0
        elif action_feature == 1:
            rel_dir_bin = 2
            action_left, action_forward, action_right = 0.0, 0.0, 1.0

        src_dist_to_target = None
        dst_dist_to_target = None
        if distance_map is not None:
            sdist = distance_map[handle, src_pos[0], src_pos[1], src_dir]
            if np.isfinite(sdist):
                src_dist_to_target = float(sdist)
            ddist = distance_map[handle, dst_pos[0], dst_pos[1], dst_dir]
            if np.isfinite(ddist):
                dst_dist_to_target = float(ddist)

        improves_over_current = False
        if src_dist_to_target is not None and dst_dist_to_target is not None:
            improves_over_current = bool(dst_dist_to_target < src_dist_to_target)

        src_transitions = self._rail_get_transitions(src_pos, src_dir)
        src_choices = max(1, int(fast_count_nonzero(src_transitions)))
        branch_choice_prob = 1.0 / float(src_choices)

        same_dir_handles = corridor.get("same_dir_handles", [])
        oncoming_handles = corridor.get("oncoming_handles", [])
        forward_handles = corridor.get("forward_handles", same_dir_handles)
        backward_handles = corridor.get("backward_handles", oncoming_handles)
        edge_agents = corridor.get("edge_agents", [])
        edge_cells = [tuple(cell) for cell, _ in edge_path]
        if edge_cells and edge_cells[-1] == tuple(dst_pos):
            edge_cells = edge_cells[:-1]

        return {
            "src": src_idx,
            "dst": dst_idx,
            "len": int(corridor.get("edge_len", 0)),
            "src_pos": (int(src_pos[0]), int(src_pos[1])),
            "dst_pos": (int(dst_pos[0]), int(dst_pos[1])),
            "src_depth": int(src_depth),
            "agents": list(edge_agents),
            "has_same_dir_agent": bool(len(same_dir_handles) > 0),
            "has_other_dir_agent": bool(len(oncoming_handles) > 0),
            "same_dir_agent_handles": list(same_dir_handles),
            "oncoming_agent_handles": list(oncoming_handles),
            "forward_agent_handles": list(forward_handles),
            "backward_agent_handles": list(backward_handles),
            "cells": edge_cells,
            "min_dist_to_target": corridor.get("min_dist_to_target", None),
            "target_on_edge": bool(corridor.get("target_on_edge", False)),
            "action": action_feature,
            "rel_dir_bin": int(rel_dir_bin),
            "action_left": float(action_left),
            "action_forward": float(action_forward),
            "action_right": float(action_right),
            "has_agents_on_edge": bool(len(edge_agents) > 0),
            "has_oncoming_edge": bool(len(oncoming_handles) > 0),
            "agents_on_edge_count": int(len(edge_agents)),
            "edge_len_cells": int(corridor.get("edge_len", 0)),
            "src_dist_to_target": src_dist_to_target,
            "dst_dist_to_target": dst_dist_to_target,
            "improves_over_current": bool(improves_over_current),
            "branch_choice_prob": float(branch_choice_prob),
        }

    def _recursive_expand_local_tree(
        self,
        handle,
        src_idx,
        depth,
        max_depth,
        max_nodes,
        agent_target,
        distance_map,
        state,
        expansion_guard,
    ):
        if depth >= max_depth or state["n_nodes"] >= max_nodes:
            return

        src_state = state["node_pos_dir_map"].get(src_idx)
        if src_state is None:
            return
        src_pos, src_dir = src_state

        guard_key = (int(src_idx), int(depth))
        if guard_key in expansion_guard:
            return
        expansion_guard.add(guard_key)

        transitions = self._rail_get_transitions(src_pos, src_dir)
        for first_dir in range(4):
            if not transitions[first_dir]:
                continue

            corridor = self._walk_corridor_until_decision(
                handle=handle,
                src_pos=src_pos,
                src_dir=src_dir,
                first_dir=first_dir,
                agent_target=agent_target,
                distance_map=distance_map,
            )
            if not corridor.get("complete", False):
                continue

            dst_pos = corridor.get("dst_pos")
            dst_dir = corridor.get("dst_dir")
            dst_type = corridor.get("dst_type")
            if dst_pos is None or dst_dir is None or dst_type is None:
                continue

            dst_key = (tuple(dst_pos), int(dst_dir))
            if dst_key not in state["node_map"]:
                if state["n_nodes"] >= max_nodes:
                    continue
                state["nodes"].append(
                    self._build_local_node_payload(
                        handle=handle,
                        ntype=dst_type,
                        npos=dst_pos,
                        ndir=dst_dir,
                        depth_value=depth + 1,
                        max_depth=max_depth,
                    )
                )
                dst_idx = state["n_nodes"]
                state["node_map"][dst_key] = dst_idx
                state["node_pos_dir_map"][dst_idx] = (tuple(dst_pos), int(dst_dir))
                state["n_nodes"] += 1
            else:
                dst_idx = state["node_map"][dst_key]

            edge_payload = self._build_corridor_edge_payload(
                handle=handle,
                src_idx=src_idx,
                dst_idx=dst_idx,
                src_pos=src_pos,
                src_dir=src_dir,
                src_depth=state["nodes"][src_idx].get("depth", depth),
                dst_pos=dst_pos,
                dst_dir=dst_dir,
                corridor=corridor,
                distance_map=distance_map,
            )
            edge_key = (
                int(edge_payload.get("src", -1)),
                int(edge_payload.get("dst", -1)),
                int(edge_payload.get("rel_dir_bin", -1)),
            )
            if edge_key not in state["edge_keys"]:
                state["edge_keys"].add(edge_key)
                state["edges"].append(edge_payload)
                state["seen_agents"].update(edge_payload.get("agents", []))

            if dst_idx in state["active_stack"]:
                continue
            state["active_stack"].add(dst_idx)
            self._recursive_expand_local_tree(
                handle=handle,
                src_idx=dst_idx,
                depth=depth + 1,
                max_depth=max_depth,
                max_nodes=max_nodes,
                agent_target=agent_target,
                distance_map=distance_map,
                state=state,
                expansion_guard=expansion_guard,
            )
            state["active_stack"].remove(dst_idx)

    def _export_debug_tree_overlay(self, handle, root_pos, tree_payload):
        prof_active = bool(self.obs_func_profile_enabled and self._obs_profile_active)
        t0 = time.perf_counter() if prof_active else 0.0

        if handle != 0 or not bool(getattr(self, "debug_tree_overlay_enabled", False)):
            return

        if len(self.env.agents) < 5:
            return

        if not hasattr(self.env, "dev_obs_dict") or self.env.dev_obs_dict is None:
            self.env.dev_obs_dict = {}

        overlay = {
            0: set(),
            1: set(),
            2: set(),
            3: set(),
            4: set(),
        }
        nodes = tree_payload.get("nodes", [])
        edges = tree_payload.get("edges", [])
        edges_by_src = {}
        for edge in edges:
            src_idx = int(edge.get("src", -1))
            if src_idx < 0:
                continue
            edges_by_src.setdefault(src_idx, []).append(edge)

        visited_nodes = set()
        visited_edges = set()

        def _node_cells(node_payload):
            node_cells = node_payload.get("cells")
            if node_cells:
                return [
                    (int(cell[0]), int(cell[1]))
                    for cell in node_cells
                    if cell is not None
                ]
            node_pos = node_payload.get("position")
            if node_pos is None:
                return []
            return [(int(node_pos[0]), int(node_pos[1]))]

        def _visit_node(node_idx):
            if node_idx in visited_nodes or node_idx < 0 or node_idx >= len(nodes):
                return
            visited_nodes.add(node_idx)

            node = nodes[node_idx]
            node_cells = _node_cells(node)
            overlay[0].update(node_cells)
            if bool(node.get("is_pre_merge", False)):
                overlay[1].update(node_cells)
            if bool(node.get("is_switch", False)):
                overlay[2].update(node_cells)

            for edge in edges_by_src.get(node_idx, []):
                edge_id = (
                    int(edge.get("src", -1)),
                    int(edge.get("dst", -1)),
                    int(edge.get("rel_dir_bin", -1)),
                )
                if edge_id in visited_edges:
                    continue
                visited_edges.add(edge_id)

                edge_depth = int(edge.get("src_depth", node.get("depth", 0)))
                pseudo_handle = 3 if (edge_depth % 2 == 0) else 4
                corridor_cells = [
                    (int(cell[0]), int(cell[1]))
                    for cell in edge.get("cells", [])
                    if cell is not None
                ]
                overlay[pseudo_handle].update(corridor_cells)

                dst_idx = int(edge.get("dst", -1))
                if dst_idx >= 0:
                    _visit_node(dst_idx)

        if nodes:
            _visit_node(0)
        elif root_pos is not None:
            overlay[0].add((int(root_pos[0]), int(root_pos[1])))

        for node in nodes:
            for cell in _node_cells(node):
                overlay[3].discard(cell)
                overlay[4].discard(cell)

        # Keep node markers visually clean: a cell should not appear as both
        # corridor parity and explicit switch/merge overlay in the same frame.
        overlay[3].difference_update(overlay[1])
        overlay[3].difference_update(overlay[2])
        overlay[4].difference_update(overlay[1])
        overlay[4].difference_update(overlay[2])

        for pseudo_handle, cells in overlay.items():
            self.env.dev_obs_dict[pseudo_handle] = set(cells)

        if prof_active:
            self._obs_prof_add('debug_overlay', time.perf_counter() - t0)

    def _local_search(self, handle, start_pos, start_dir, depth_limit):
        t_start = time.perf_counter()
        """Build local decision-point tree recursively from corridor building blocks."""
        if start_pos is None or start_dir is None or self.env is None or self.env.rail is None:
            return {"nodes": [], "edges": [], "seen_agents": []}

        agent = self.env.agents[handle]
        agent_target = agent.target
        distance_map = self.env.distance_map.get()
        max_nodes = int(getattr(self, "local_search_max_nodes", 72))
        max_depth = int(depth_limit)

        # Early exit für Waiting/Done
        if agent.state in [TrainState.WAITING, TrainState.DONE]:
            return {"nodes": [], "edges": [], "seen_agents": []}


        # Initiale Position und Richtung bestimmen
        if agent.position is not None:
            pos = agent.position
            direction = agent.direction
        elif agent.state == TrainState.READY_TO_DEPART:
            pos = agent.initial_position
            direction = agent.initial_direction
        else:
            return {"nodes": [], "edges": [], "seen_agents": []}


        state = {
            "nodes": [],
            "edges": [],
            "edge_keys": set(),
            "seen_agents": set(),
            "node_map": {},
            "node_pos_dir_map": {},
            "active_stack": {0},
            "n_nodes": 0,
        }

        state["nodes"].append(
            self._build_local_node_payload(
                handle=handle,
                ntype=0,
                npos=pos,
                ndir=direction,
                depth_value=0,
                max_depth=max_depth,
            )
        )
        state["node_map"][(tuple(pos), int(direction))] = 0
        state["node_pos_dir_map"][0] = (tuple(pos), int(direction))
        state["n_nodes"] = 1

        self._recursive_expand_local_tree(
            handle=handle,
            src_idx=0,
            depth=0,
            max_depth=max_depth,
            max_nodes=max_nodes,
            agent_target=agent_target,
            distance_map=distance_map,
            state=state,
            expansion_guard=set(),
        )

        if self._obs_profile_active:
            self._obs_prof_add("local_search", time.perf_counter() - t_start)
        return {
            "nodes": state["nodes"],
            "edges": state["edges"],
            "seen_agents": sorted(state["seen_agents"]),
        }

    @staticmethod
    def _depth_to_proximity(depth_value: int, max_depth: int) -> float:
        if depth_value is None or int(depth_value) < 0:
            return 0.0
        denom = max(1, int(max_depth))
        # 1.0 means immediate deadlock, 0.0 means no deadlock in probe horizon.
        return float(np.clip(1.0 - (float(depth_value) / float(denom)), 0.0, 1.0))

    def _calculate_deadlock_profile(self, handle, pos, direction, max_depth=14, max_states=256, transition_cache=None):
        """Compute local deadlock profile with risk and explicit distance-to-deadlock signals."""
        prof_active = bool(self.obs_func_profile_enabled and self._obs_profile_active)
        t0 = time.perf_counter() if prof_active else 0.0
        if pos is None or direction is None or self.env is None or self.env.rail is None:
            raise ValueError(f"_calculate_deadlock_profile received invalid inputs for agent {handle}")

        if transition_cache is None:
            transition_cache = {}

        def _get_transitions_cached(cell_pos, cell_dir):
            key = (int(cell_pos[0]), int(cell_pos[1]), int(cell_dir))
            if key in transition_cache:
                return transition_cache[key]
            trans = self._rail_get_transitions(cell_pos, cell_dir)
            transition_cache[key] = trans
            return trans

        visited = set()
        frontier = [(pos, direction, 0)]
        deadlock_risk = 0.0
        min_deadlock_depth = None
        min_hard_block_depth = None
        min_soft_block_depth = None
        min_merge_conflict_depth = None
        while frontier:
            if len(visited) >= max(8, int(max_states)):
                break
            current_pos, current_dir, depth = frontier.pop()
            if depth > max(1, int(max_depth)):
                continue
            visited_key = (self._pos_tuple(current_pos), int(current_dir))
            if visited_key in visited:
                continue
            visited.add(visited_key)
            transitions = _get_transitions_cached(current_pos, current_dir)
            num_transitions = fast_count_nonzero(transitions)
            in_deg = self._incoming_degree(current_pos, transition_cache=transition_cache)

            has_oncoming = False
            if self.agent_map is not None:
                occ = self._agent_at_pos(current_pos)
                if occ != -1 and occ != handle:
                    occ_dir = self.env.agents[occ].direction
                    if occ_dir is not None and DecisionPointUtils.is_opposite_direction(current_dir, occ_dir):
                        has_oncoming = True

            hard_block = bool(num_transitions == 0)
            soft_block = bool(has_oncoming and num_transitions <= 1)
            merge_conflict = bool(in_deg > 2 and num_transitions == 1 and depth <= 6)

            if hard_block or soft_block:
                if min_deadlock_depth is None or int(depth) < int(min_deadlock_depth):
                    min_deadlock_depth = int(depth)
            if hard_block:
                if min_hard_block_depth is None or int(depth) < int(min_hard_block_depth):
                    min_hard_block_depth = int(depth)
            if soft_block:
                if min_soft_block_depth is None or int(depth) < int(min_soft_block_depth):
                    min_soft_block_depth = int(depth)
            if merge_conflict:
                if min_merge_conflict_depth is None or int(depth) < int(min_merge_conflict_depth):
                    min_merge_conflict_depth = int(depth)

            if hard_block:
                deadlock_risk += 1.0
            elif soft_block:
                deadlock_risk += 0.65
            elif merge_conflict:
                deadlock_risk += 0.30
            elif has_oncoming:
                deadlock_risk += 0.18
            elif num_transitions <= 1:
                deadlock_risk += 0.03

            for next_dir in range(4):
                if transitions[next_dir]:
                    next_pos = get_new_position(current_pos, next_dir)
                    frontier.append((next_pos, next_dir, depth + 1))

        probe_depth = max(1, int(max_depth))
        risk_norm = max(4.0, min(24.0, float(len(visited)) * 0.35))
        effective_hard_depth = min_hard_block_depth
        if effective_hard_depth is None:
            effective_hard_depth = min_soft_block_depth
        result = {
            "risk": float(min(deadlock_risk / risk_norm, 1.0)),
            "min_deadlock_depth": int(min_deadlock_depth) if min_deadlock_depth is not None else -1,
            "min_hard_block_depth": int(effective_hard_depth) if effective_hard_depth is not None else -1,
            "min_soft_block_depth": int(min_soft_block_depth) if min_soft_block_depth is not None else -1,
            "min_merge_conflict_depth": int(min_merge_conflict_depth) if min_merge_conflict_depth is not None else -1,
            "deadlock_distance_norm": float(self._depth_to_proximity(min_deadlock_depth, probe_depth)),
            "hard_block_distance_norm": float(self._depth_to_proximity(effective_hard_depth, probe_depth)),
            "soft_block_distance_norm": float(self._depth_to_proximity(min_soft_block_depth, probe_depth)),
            "merge_conflict_distance_norm": float(self._depth_to_proximity(min_merge_conflict_depth, probe_depth)),
        }
        if prof_active:
            self._obs_prof_add('deadlock_profile', time.perf_counter() - t0)
        return result

    @staticmethod
    def getObservationSize() -> int:
        return DecisionPointObservation.OBS_SIZE

    @classmethod
    def _print_feature_layout_doc(cls):
        if os.getenv("DEBUG_OBSERVATION", "0") == "1":
            print(">> DecisionPointObservation (15D Base + Tree Payload) - Feature-Layout:")
            for idx, name, desc in cls.FEATURE_GROUPS_DOC:
                print(f"   {idx:<8} {name:<18} {desc}")
            print("   0..14    base_feature_specs  exact index-to-meaning mapping (15D total)")
            print("   Tree Payload: nodes and edges contain deadlock_risk, deadlock_ahead, deadlock_hard_block")

    @classmethod
    def _cleanup_base_features(cls, raw_features: np.ndarray) -> None:
        # No masking: all 15 base features are kept as-is.
        return

    @staticmethod
    def _encode_detect_deadlock(raw: float) -> float:
        return 1.0 if raw > 0 else 0.0

    @staticmethod
    def _encode_deadlock_signal(deadlock_distance: float) -> float:
        if deadlock_distance is None or deadlock_distance <= 0:
            return 0.0
        # Steeper decay: nearby deadlocks become more prominent, which helps
        # the policy separate "slightly risky" from "immediate danger".
        return min(1.0, 1.0 / (1.0 + deadlock_distance / 2.5))

    @staticmethod
    def _cell_type_index_from_decision_type(decision_type: int) -> int:
        if decision_type & 8:
            return 4
        if decision_type == 1:
            return 0
        if decision_type & 2:
            return 3
        if decision_type & 4:
            return 2
        return 1

    def _is_switch_at_current_cell(self, pos, direction) -> bool:
        """True if the agent stands on a switching cell right now."""
        transitions = self._rail_get_transitions(pos, direction)
        return fast_count_nonzero(transitions) > 1

    def _incoming_degree(self, cell_pos, transition_cache=None) -> int:
        """Count incoming directed edges to a cell by local 4-neighborhood scan."""
        if transition_cache is None:
            transition_cache = {}

        def _get_transitions_cached(pos, direction):
            key = (int(pos[0]), int(pos[1]), int(direction))
            if key in transition_cache:
                return transition_cache[key]
            trans = self._rail_get_transitions(pos, direction)
            transition_cache[key] = trans
            return trans

        incoming_edges = set()
        for prev_dir in range(4):
            prev_pos = get_new_position(cell_pos, (prev_dir + 2) % 4)
            if prev_pos[0] < 0 or prev_pos[0] >= self.env.height or prev_pos[1] < 0 or prev_pos[1] >= self.env.width:
                continue
            for d in range(4):
                trans = _get_transitions_cached(prev_pos, d)
                for nd in range(4):
                    if not trans[nd]:
                        continue
                    if get_new_position(prev_pos, nd) == cell_pos:
                        incoming_edges.add((prev_pos[0], prev_pos[1], d, nd))
        return len(incoming_edges)

    def _incoming_agent_handles(self, cell_pos, handle_exclude: int, transition_cache=None) -> list:
        """Collect agents that can enter cell_pos through an incoming directed edge."""
        if self.agent_map is None:
            return []

        if transition_cache is None:
            transition_cache = {}

        def _get_transitions_cached(pos, direction):
            key = (int(pos[0]), int(pos[1]), int(direction))
            if key in transition_cache:
                return transition_cache[key]
            trans = self._rail_get_transitions(pos, direction)
            transition_cache[key] = trans
            return trans

        found = set()
        for prev_dir in range(4):
            prev_pos = get_new_position(cell_pos, (prev_dir + 2) % 4)
            if prev_pos[0] < 0 or prev_pos[0] >= self.env.height or prev_pos[1] < 0 or prev_pos[1] >= self.env.width:
                continue
            agent_idx = self._agent_at_pos(prev_pos)
            if agent_idx == -1 or agent_idx == handle_exclude:
                continue
            a_dir = self.env.agents[agent_idx].direction
            if a_dir is None:
                continue
            trans = _get_transitions_cached(prev_pos, a_dir)
            for nd in range(4):
                if not trans[nd]:
                    continue
                if get_new_position(prev_pos, nd) == cell_pos:
                    found.add(agent_idx)
                    break
        return sorted(found)

    def _is_pre_merge_one_exit(self, pos, direction, transitions) -> bool:
        """True if agent is exactly one step before a merge/conflict node with one current exit.

        Semantics for DAG-style routing:
        - current cell: exactly one usable outgoing edge for the current heading
        - next cell: true merge node, i.e. receives multiple incoming edges and
          has a single onward edge for the arriving orientation
        """
        if fast_count_nonzero(transitions) != 1:
            return False
        ndir = int(fast_argmax(transitions))
        if not transitions[ndir]:
            return False
        next_pos = get_new_position(pos, ndir)
        if next_pos[0] < 0 or next_pos[0] >= self.env.height or next_pos[1] < 0 or next_pos[1] >= self.env.width:
            return False
        in_deg = self._incoming_degree(next_pos)
        if in_deg <= 1:
            return False
        next_transitions_arrival = self._rail_get_transitions(next_pos, ndir)
        return fast_count_nonzero(next_transitions_arrival) == 1

    def _decision_type_at_position(self, pos, direction, target) -> int:
        if pos == target:
            return 8
        transitions = self._rail_get_transitions(pos, direction)
        decision_type = 0
        if self._is_switch_at_current_cell(pos, direction):
            decision_type += 2
        if self._is_pre_merge_one_exit(pos, direction, transitions):
            decision_type += 4
        return decision_type

    def _build_base_features(self, handle, agent, pos, direction, distance_map):
        prof_active = bool(self.obs_func_profile_enabled and self._obs_profile_active)
        t0 = time.perf_counter() if prof_active else 0.0

        raw_features = np.zeros(self.BASE_OBS_SIZE, dtype=np.float32)

        transitions = self._rail_get_transitions(pos, direction)
        left_dir = (int(direction) - 1) % 4
        fwd_dir = int(direction) % 4
        right_dir = (int(direction) + 1) % 4

        raw_features[0] = 1.0 if transitions[left_dir] else 0.0
        raw_features[1] = 1.0 if transitions[fwd_dir] else 0.0
        raw_features[2] = 1.0 if transitions[right_dir] else 0.0

        successor_dist = {}
        for ndir in (left_dir, fwd_dir, right_dir):
            if transitions[ndir]:
                npos = get_new_position(pos, ndir)
                successor_dist[ndir] = self._safe_distance(handle, npos, ndir, distance_map)

        finite_successors = [d for d in successor_dist.values() if np.isfinite(d)]
        best_successor_dist = min(finite_successors) if finite_successors else np.inf

        for feat_idx, ndir in ((3, left_dir), (4, fwd_dir), (5, right_dir)):
            ndist = successor_dist.get(ndir, np.inf)
            if np.isfinite(best_successor_dist) and np.isfinite(ndist):
                # Relative quality against the best local successor:
                # 0.0 for best branch, negative for longer alternatives.
                # Use smooth exponential squashing instead of hard clipping so
                # large gaps remain distinguishable but bounded.
                raw_gap = float(best_successor_dist - ndist)
                raw_features[feat_idx] = self._exp_squash_signed(raw_gap)
            else:
                # Mark unavailable/unreachable branches as clearly worse than
                # the best local successor to avoid conflicting with SP hints.
                raw_features[feat_idx] = -1.0

        all_distance = []
        self_distance = np.inf
        for idx, a in enumerate(self.env.agents):
            apos = a.position if a.position is not None else a.initial_position
            adir = a.direction if a.direction is not None else a.initial_direction
            if apos is None or adir is None:
                adist = np.inf
            else:
                adist = float(distance_map[a.handle, apos[0], apos[1], adir])
            all_distance.append((a.handle, adist, idx))
            if a.handle == handle:
                self_distance = adist

        all_distance.sort(key=lambda x: (x[1], x[2]))
        finite_distances = [dist for _, dist, _ in all_distance if np.isfinite(dist)]
        finite_unique = sorted(set(finite_distances))

        if len(finite_unique) >= 2:
            # Generic normalized distance-rank priority in [0, 1].
            # Best (smallest remaining distance) -> 0.0, worst -> 1.0.
            value_to_rank = {dist: i for i, dist in enumerate(finite_unique)}
            denom = max(1, len(finite_unique) - 1)
            rank = value_to_rank.get(self_distance, len(finite_unique) - 1)
            priority_rank = float(rank) / float(denom)
        else:
            # If cohort rank is undefined (all same distance / single sample),
            # fall back to normalized remaining distance so the feature remains informative.
            fallback_max = max(1.0, float(self.env.width + self.env.height))
            priority_rank = self._distance_to_unit(self_distance, fallback_max)

        is_started = agent.position is not None
        # Export selected lifecycle flags used by the current 15D contract.
        # st_3=READY_TO_DEPART, st_4=MALFUNCTION, st_6=not-started.
        raw_features[6] = 1.0 if (not is_started) and agent.state == TrainState.READY_TO_DEPART else 0.0  # st_3 (READY_TO_DEPART)
        raw_features[7] = 1.0 if is_started and agent.state == TrainState.MALFUNCTION else 0.0  # st_4 (MALFUNCTION)
        raw_features[8] = 0.0 if is_started else 1.0                       # st_6 (not-started)
        raw_features[9] = priority_rank

        raw_features[10] = 1.0 if self._is_pre_merge_one_exit(pos, direction, transitions) else 0.0
        raw_features[11] = 1.0 if self._is_switch_at_current_cell(pos, direction) else 0.0

        sp_left, sp_fwd, sp_right = self._shortest_path_action_hint(
            handle=handle,
            pos=pos,
            direction=direction,
            transitions=transitions,
            distance_map=distance_map,
        )
        raw_features[12] = float(sp_left)
        raw_features[13] = float(sp_fwd)
        raw_features[14] = float(sp_right)

        if prof_active:
            self._obs_prof_add('base_features', time.perf_counter() - t0)

        return raw_features

    def get(self, handle: int = 0):
        """Return (base_features, seen_agents, raw_tree_payload) for one agent.
        Export 15 base features (dead TrainStates removed, deadlock moved to tree).
        Deadlock information is embedded in tree payload nodes/edges.
        """
        prof_active = False
        t0 = 0.0
        if self.obs_func_profile_enabled:
            prof_active = (self._obs_func_profile_call_idx % self.obs_func_profile_sample_every) == 0
            self._obs_func_profile_call_idx += 1
            if prof_active:
                t0 = time.perf_counter()

        agent = self.env.agents[handle]
        pos = agent.position if agent.position is not None else agent.initial_position
        direction = agent.direction if agent.direction is not None else agent.initial_direction
        target = agent.target
        if pos is None or target is None or direction is None:
            raise ValueError(f"Agent {handle} has invalid start data for observation building")
        distance_map = self.env.distance_map.get()

        # Lokale Suche → Baum-Payload für trainierbare Encoder-Integration
        search_depth = max(int(self.search_depth), int(getattr(self, "local_search_min_search_depth", 8)))
        prev_active = self._obs_profile_active
        self._obs_profile_active = bool(prof_active)
        tree_payload = self._local_search(handle, pos, direction, search_depth)
        self._obs_profile_active = prev_active
        local_search_seen_agents = set(tree_payload.get("seen_agents", []))

        # Keep structured tree context available for downstream temporal wrappers.
        if not hasattr(self.env, "dev_tree_dict"):
            self.env.dev_tree_dict = {}
        self.env.dev_tree_dict[handle] = tree_payload
        self._export_debug_tree_overlay(handle, pos, tree_payload)

        opp_agents = set()
        opp_agents.update(local_search_seen_agents)
        for other in self.env.agents:
            if other.handle == handle:
                continue
            other_pos = other.position if other.position is not None else other.initial_position
            if other_pos == pos:
                opp_agents.add(other.handle)

        # Fill base features according to the 15D schema.
        raw_features = self._build_base_features(
            handle=handle,
            agent=agent,
            pos=pos,
            direction=direction,
            distance_map=distance_map,
        )

        # No masking: all features are exported
        base_features = raw_features.copy() 

        agent.cur_opp_agent_handles = sorted(opp_agents)
        if prof_active:
            self._obs_prof_add('get', time.perf_counter() - t0)
        return (base_features, agent.cur_opp_agent_handles, tree_payload)

    def get_many(self, handles: list = None, is_end_of_episode: bool = False, episode_count: int = None):
        t0_many = time.perf_counter() if self.obs_func_profile_enabled else 0.0
        # Nur noch für Rückwärtskompatibilität: Counter bleibt, aber nicht mehr für Ausgabe genutzt
        type(self)._get_many_call_count += 1
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
            if not hasattr(agent, 'opp_agent_handles'):
                agent.opp_agent_handles = []
            if not hasattr(agent, 'cur_opp_agent_handles'):
                agent.cur_opp_agent_handles = []

        result = []
        all_features = []
        tree_stats = []
        for handle in handles:
            entry = self.get(handle)
            result.append(entry)
            all_features.append(entry[0])
            tree = entry[2]
            n_nodes = len(tree.get("nodes", []))
            n_edges = len(tree.get("edges", []))
            seen_agents = tree.get("seen_agents", [])
            tree_stats.append((handle, n_nodes, n_edges, seen_agents))

        # --- Statistik der letzten 100 Episoden sammeln ---
        if len(all_features) > 0:
            arr = np.stack(all_features, axis=0)
            # Ringpuffer für Features
            if not hasattr(type(self), '_last_100_features'):
                type(self)._last_100_features = []
            if not hasattr(type(self), '_last_100_tree_stats'):
                type(self)._last_100_tree_stats = []
            type(self)._last_100_features.append(arr)
            type(self)._last_100_tree_stats.append(tree_stats)
            if len(type(self)._last_100_features) > 100:
                type(self)._last_100_features.pop(0)
            if len(type(self)._last_100_tree_stats) > 100:
                type(self)._last_100_tree_stats.pop(0)

        # --- Ausgabe nur am Ende jeder 50. Episode ---
        if is_end_of_episode and episode_count is not None and episode_count > 0 and episode_count % 50 == 0:
            feature_names = [name for _, name, _ in type(self).BASE_FEATURE_SPECS]
            last_feats = type(self)._last_100_features
            last_trees = type(self)._last_100_tree_stats
            all_feats = np.concatenate(last_feats, axis=0) if last_feats else arr
            n_ep = len(last_feats)

            # ── helpers ──────────────────────────────────────────────────────
            def _trend_label(series: np.ndarray) -> str:
                """Return monotone-trend label for a 1-D time series of episode means."""
                if len(series) < 4:
                    return "n/a"
                diffs = np.diff(series.astype(float))
                n_down = int(np.sum(diffs < -1e-4))
                n_up   = int(np.sum(diffs >  1e-4))
                frac_down = n_down / max(1, len(diffs))
                frac_up   = n_up   / max(1, len(diffs))
                if frac_down >= 0.65:
                    return "↓ mono-fall"
                if frac_up >= 0.65:
                    return "↑ mono-rise"
                if frac_down >= 0.40 and frac_up < 0.20:
                    return "↓ tend-fall"
                if frac_up >= 0.40 and frac_down < 0.20:
                    return "↑ tend-rise"
                return "↔ flat/noisy"

            def _ep_means(feat_idx: int) -> np.ndarray:
                """Per-episode mean of a feature over all agents/steps in that episode."""
                return np.array([ep[:, feat_idx].mean() for ep in last_feats if ep.shape[0] > 0])

            def _deadlock_ratio(col: np.ndarray) -> float:
                """Fraction of steps where a binary feature is active (value > 0.5)."""
                return float(np.mean(col > 0.5)) if len(col) > 0 else 0.0

            # ── header ───────────────────────────────────────────────────────
            W = "=" * 72
            print(f"\n{W}")
            print(f"  [DecisionPointObs] BASE FEATURE REPORT  (n={n_ep} episodes, ep={episode_count})")
            print(W)
            print(f"  {'#':>2}  {'Feature':<22} {'min':>6} {'max':>6} {'mean':>7} {'std':>6}  {'trend':>13}  note")
            print(f"  {'-'*68}")

            for idx, name in enumerate(feature_names):
                col = all_feats[:, idx]
                ep_m = _ep_means(idx)
                trend = _trend_label(ep_m)
                note = ""
                # ── feature-specific annotations ─────────────────────────────
                if name == "priority_rank":
                    # Should fall toward 0 as agent approaches goal
                    if "fall" in trend:
                        note = "✅ agent approaching goal"
                    elif np.std(col) < 0.01:
                        note = "⚠️  CONSTANT – check obs"
                    else:
                        note = "🟡 no clear progress trend"
                elif name in ("st_3",):
                    ready_frac = float(np.mean(col > 0.5))
                    if ready_frac > 0.5:
                        note = f"✅ ready_to_depart {ready_frac*100:.0f}% of steps"
                    else:
                        note = f"🟡 ready_to_depart {ready_frac*100:.0f}% of steps"
                elif name in ("st_4",):
                    malf_frac = float(np.mean(col > 0.5))
                    note = f"malfunction={malf_frac*100:.0f}% of steps"
                elif name in ("st_6",):
                    not_started_frac = float(np.mean(col > 0.5))
                    note = f"not_started={not_started_frac*100:.0f}% of steps"
                elif name in ("sp_forward", "sp_left", "sp_right"):
                    frac = float(np.mean(col > 0.5))
                    note = f"used {frac*100:.0f}% of steps"
                elif name == "is_switch":
                    frac = float(np.mean(col > 0.5))
                    note = f"at switch {frac*100:.0f}% of steps"
                elif name == "is_pre_merge":
                    frac = float(np.mean(col > 0.5))
                    note = f"pre-merge {frac*100:.0f}% of steps"

                print(
                    f"  {idx:>2}  {name:<22} "
                    f"{np.min(col):>6.3f} {np.max(col):>6.3f} "
                    f"{np.mean(col):>7.4f} {np.std(col):>6.4f}  "
                    f"{trend:>13}  {note}"
                )

            # ── priority_rank episode-mean trend (compact) ───────────────────
            pr_ep = _ep_means(9)  # feature [9] = priority_rank
            if len(pr_ep) >= 2:
                half = max(1, len(pr_ep) // 2)
                first_half = pr_ep[:half].mean()
                second_half = pr_ep[half:].mean()
                delta = second_half - first_half
                arrow = "↓" if delta < -0.02 else ("↑" if delta > 0.02 else "↔")
                print(f"\n  priority_rank half-half: first={first_half:.4f} → second={second_half:.4f}  Δ={delta:+.4f} {arrow}")

            # ── Tree-Statistik ────────────────────────────────────────────────
            all_nodes, all_edges = [], []
            for ep_tree_stats in last_trees:
                for _h, n_nodes, n_edges, _seen in ep_tree_stats:
                    all_nodes.append(n_nodes)
                    all_edges.append(n_edges)
            if all_nodes:
                print(f"\n  Tree: nodes={np.mean(all_nodes):.1f}±{np.std(all_nodes):.1f} "
                      f"[{np.min(all_nodes)},{np.max(all_nodes)}]   "
                      f"edges={np.mean(all_edges):.1f}±{np.std(all_edges):.1f} "
                      f"[{np.min(all_edges)},{np.max(all_edges)}]")
            print(W)

        if self.obs_func_profile_enabled:
            self._obs_prof_add('get_many', time.perf_counter() - t0_many)
            if is_end_of_episode and episode_count is not None and (episode_count + 1) % self.obs_func_profile_interval == 0:
                g = self._obs_func_prof
                get_mean_ms = (1000.0 * g['get']['sum'] / g['get']['count']) if g['get']['count'] > 0 else 0.0
                gm_mean_ms = (1000.0 * g['get_many']['sum'] / g['get_many']['count']) if g['get_many']['count'] > 0 else 0.0
                ls_mean_ms = (1000.0 * g['local_search']['sum'] / g['local_search']['count']) if g['local_search']['count'] > 0 else 0.0
                dl_mean_ms = (1000.0 * g['deadlock_profile']['sum'] / g['deadlock_profile']['count']) if g['deadlock_profile']['count'] > 0 else 0.0
                bf_mean_ms = (1000.0 * g['base_features']['sum'] / g['base_features']['count']) if g['base_features']['count'] > 0 else 0.0
                ov_mean_ms = (1000.0 * g['debug_overlay']['sum'] / g['debug_overlay']['count']) if g['debug_overlay']['count'] > 0 else 0.0
                dl_per_ls = (float(g['deadlock_profile']['count']) / float(max(1, g['local_search']['count']))) if g['local_search']['count'] > 0 else 0.0
                print(
                    f"[ObsFnPerf] ep={episode_count + 1} interval={self.obs_func_profile_interval} "
                    f"sample_every={self.obs_func_profile_sample_every} "
                    f"get={get_mean_ms:.3f}ms get_many={gm_mean_ms:.3f}ms "
                    f"local_search={ls_mean_ms:.3f}ms deadlock_profile={dl_mean_ms:.3f}ms "
                    f"base_features={bf_mean_ms:.3f}ms debug_overlay={ov_mean_ms:.3f}ms "
                    f"deadlock_calls_per_local_search={dl_per_ls:.2f}"
                )
                type(self)._last_obs_fn_perf_report = {
                    'episode': int(episode_count + 1),
                    'interval': int(self.obs_func_profile_interval),
                    'sample_every': int(self.obs_func_profile_sample_every),
                    'get_mean_ms': float(get_mean_ms),
                    'get_many_mean_ms': float(gm_mean_ms),
                    'local_search_mean_ms': float(ls_mean_ms),
                    'deadlock_profile_mean_ms': float(dl_mean_ms),
                    'base_features_mean_ms': float(bf_mean_ms),
                    'debug_overlay_mean_ms': float(ov_mean_ms),
                    'deadlock_calls_per_local_search': float(dl_per_ls),
                }
                for bucket in g.values():
                    bucket['sum'] = 0.0
                    bucket['count'] = 0

        for agent in self.env.agents:
            agent.opp_agent_handles = agent.cur_opp_agent_handles
        return result

    @staticmethod
    def _normalise_distance(value: float, max_dist: float) -> float:
        if value is None or value == _UNREACHABLE:
            return 0.0
        if not np.isfinite(value):
            return 0.0
        if max_dist <= 0:
            return 0.0
        return float(np.clip(value / max_dist, 0.0, 1.0))

    @staticmethod
    def _normalise_count(value: float) -> float:
        if value is None or value < 0:
            return 0.0
        return float(value) / (float(value) + 8.0)

    def _shortest_path_action_hint(self, handle, pos, direction, transitions, distance_map):
        """Compute which direction (L/F/R) is best according to distance map."""
        best_hint = [0.0, 0.0, 0.0]
        min_dist = np.inf
        best_idx = None
        for idx, rel in enumerate((-1, 0, 1)):
            ndir = (direction + rel) % 4
            if transitions[ndir]:
                npos = get_new_position(pos, ndir)
                dist = distance_map[handle, npos[0], npos[1], ndir]
                if np.isfinite(dist) and dist < min_dist:
                    min_dist = dist
                    best_idx = idx
        if best_idx is not None:
            best_hint[best_idx] = 1.0
        return best_hint

    def _detect_deadlock(self, handle, pos, direction):
        """Detect confirmed corridor blockage before the next switch."""
        return DecisionPointUtils.detect_corridor_blockage(
            self.env,
            self.agent_map,
            handle,
            pos,
            direction,
            {handle},
            16,
            0,
        )
