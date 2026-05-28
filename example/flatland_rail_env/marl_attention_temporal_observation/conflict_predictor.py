"""
ConflictPredictor — CBS-inspired multi-agent trajectory predictor.

Provenance:
- Cooperative A* / time-space reservation tables (Silver 2005,
  https://www.cs.ualberta.ca/~mmueller/ps/2005/Silver-AIIDE05.pdf)
- Conflict-Based Search (CBS, Sharon et al. 2015,
  https://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/view/5062)
- Local conflict awareness for MARL (PRIMAL2, Damani et al. 2021,
  https://arxiv.org/abs/2010.08364)

Design:
For each agent we project a K-step trajectory by greedily following the
shortest-path direction (using env.distance_map). All trajectories are
stored in a time-space reservation table that supports fast queries:
  - vertex_conflict_horizon: when does another agent enter cell (r, c)?
  - edge_conflict: head-on swap with another agent?
  - lane_agent_count: how many other agents in this corridor?
  - global_conflict_pressure: overall conflict density
  - my_priority_global: my rank by remaining shortest-path distance
  - my_sp_blocked_score: how blocked is my SP by others?
  - expected_yield_count: how many agents benefit if I stop now?

Updates: ONCE per env-step. Cached via _last_update_step.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_argmax, fast_count_nonzero
from flatland.envs.step_utils.states import TrainState

from marl_attention_temporal_observation.decision_point_utils import DecisionPointUtils


class ConflictPredictor:
    """K-step trajectory predictor with CBS-style conflict queries."""

    def __init__(self, env, horizon: int = 5):
        self.env = env
        self.horizon = int(horizon)

        # ── Time-space reservation tables ─────────────────────────────────
        # (r, c, t) → list of agent handles that occupy this vertex at time t
        self._vertex_res: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
        # (r1, c1, r2, c2, t) → list of handles that traverse this edge at t
        self._edge_res: Dict[Tuple[int, int, int, int, int], List[int]] = defaultdict(list)
        # handle → predicted path [(r, c), ...] of length up to horizon+1
        self._predicted_paths: Dict[int, List[Tuple[int, int]]] = {}
        # handle → remaining SP distance at start (for global priority)
        self._sp_distance_at_start: Dict[int, float] = {}

        # Cache control
        self._last_update_step: int = -1
        self._n_active_agents: int = 0
        self._n_predicted_conflicts: int = 0

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────

    def update(self, agent_map: np.ndarray) -> None:
        """Recompute predictions. Idempotent: safe to call multiple times per step.
        
        Args:
            agent_map: (H, W) np.int32 array, agent handle at each cell or -1.
                       Used to detect agents that are currently blocking each other.
        """
        current_step = int(getattr(self.env, "_elapsed_steps", 0))
        # If already updated this step → skip
        if current_step == self._last_update_step:
            return

        self._vertex_res.clear()
        self._edge_res.clear()
        self._predicted_paths.clear()
        self._sp_distance_at_start.clear()
        self._n_active_agents = 0

        distance_map = self.env.distance_map.get()

        for agent in self.env.agents:
            path = self._project_path(agent, distance_map)
            self._predicted_paths[agent.handle] = path

            if path:
                # path entries are (row, col, direction).
                # Use the direction from path[0] — guaranteed legal by
                # construction (it's the agent's actual heading or initial).
                start_r, start_c, start_dir = path[0]
                d = float(distance_map[agent.handle, start_r, start_c, int(start_dir)])

                # Fallback: at PRE_MERGE/MERGING the agent's current direction
                # may produce inf distance. Try min over all 4 dirs.
                if not np.isfinite(d):
                    all_dists = [
                        float(distance_map[agent.handle, start_r, start_c, int(dd)])
                        for dd in range(4)
                    ]
                    finite = [x for x in all_dists if np.isfinite(x)]
                    d = min(finite) if finite else float("inf")

                self._sp_distance_at_start[agent.handle] = d

                if agent.state in (TrainState.MOVING, TrainState.STOPPED):
                    self._n_active_agents += 1

                # Fill reservation tables.
                # Path entries are now (r, c, dir) — slice the (r, c) part.
                for t, entry in enumerate(path):
                    pos_r, pos_c = entry[0], entry[1]
                    self._vertex_res[(pos_r, pos_c, t)].append(agent.handle)
                    if t > 0:
                        prev_r, prev_c = path[t - 1][0], path[t - 1][1]
                        self._edge_res[
                            (prev_r, prev_c, pos_r, pos_c, t)
                        ].append(agent.handle)

        # Global conflict counter
        self._n_predicted_conflicts = sum(
            1 for users in self._vertex_res.values() if len(users) > 1
        )


        self._last_update_step = current_step

    def _project_path(self, agent, distance_map) -> List[Tuple[int, int, int]]:
        """Project agent's K-step trajectory along shortest path.

        Returns:
            List of (row, col, direction) tuples. Direction is the heading
            WITH WHICH the agent ENTERS that cell (i.e., direction[t] is the
            direction taken at step t-1 to reach pos[t]).
            For path[0] (start), direction is the agent's current/initial heading.

        Strategy:
        1. WAITING / DONE / no-direction → empty path (agent does not move).
        2. READY_TO_DEPART → use initial_position/direction.
        3. MOVING / STOPPED → use current position/direction.
        4. At each step, pick the direction that minimizes distance_map.
        5. Stop early if no valid transition or out-of-bounds.
        """
        # Filter inactive agents
        if agent.state in (TrainState.DONE,):
            return []
        if agent.state == TrainState.WAITING:
            return []

        # Pick start position/direction
        if agent.position is not None:
            pos = agent.position
            direction = agent.direction
        elif agent.state == TrainState.READY_TO_DEPART:
            pos = agent.initial_position
            direction = agent.initial_direction
        else:
            return []

        if pos is None or direction is None:
            return []

        # Path entries are (row, col, direction-arriving-here).
        # path[0] uses the agent's current heading.
        path: List[Tuple[int, int, int]] = [
            (int(pos[0]), int(pos[1]), int(direction))
        ]
        cur_pos = (int(pos[0]), int(pos[1]))
        cur_dir = int(direction)

        for _ in range(self.horizon):
            transitions = self.env.rail.get_transitions(cur_pos[0], cur_pos[1], cur_dir)
            n_trans = fast_count_nonzero(transitions)
            if n_trans == 0:
                break  # dead end

            # Choose direction with minimum distance to target
            best_dir = None
            best_dist = float("inf")
            for nd in range(4):
                if not transitions[nd]:
                    continue
                npos = get_new_position(cur_pos, nd)
                if (npos[0] < 0 or npos[0] >= self.env.height or
                        npos[1] < 0 or npos[1] >= self.env.width):
                    continue
                d = float(distance_map[agent.handle, npos[0], npos[1], nd])
                if d < best_dist:
                    best_dist = d
                    best_dir = nd

            if best_dir is None:
                break

            next_pos = get_new_position(cur_pos, best_dir)
            # NEW: store direction with which we entered this cell
            path.append((int(next_pos[0]), int(next_pos[1]), int(best_dir)))
            cur_pos = (int(next_pos[0]), int(next_pos[1]))
            cur_dir = int(best_dir)

        return path


    # ─────────────────────────────────────────────────────────────────────
    # Local conflict queries
    # ─────────────────────────────────────────────────────────────────────

    def predicted_path(self, handle: int) -> List[Tuple[int, int]]:
        """Get predicted K-step path for an agent, or empty list."""
        return self._predicted_paths.get(handle, [])

    def vertex_conflict_horizon(
        self, target_cell: Tuple[int, int], my_handle: int
    ) -> float:
        """When does another agent first enter target_cell?
        
        Returns:
            1.0  → conflict at T+1 (immediate)
            0.5  → conflict at T+(horizon/2)
            0.0  → no conflict in horizon
        """
        target = (int(target_cell[0]), int(target_cell[1]))
        for t in range(1, self.horizon + 1):
            users = self._vertex_res.get((target[0], target[1], t), [])
            others = [h for h in users if h != my_handle]
            if others:
                # Earlier conflict → higher value (more urgent)
                return float(1.0 - (t - 1) / max(1, self.horizon))
        return 0.0

    def edge_conflict(
        self,
        my_pos: Tuple[int, int],
        target_cell: Tuple[int, int],
        my_handle: int,
    ) -> float:
        """Head-on swap conflict at T+1.
        
        Returns 1.0 if another agent is predicted to move target_cell → my_pos at t=1.
        """
        my_p = (int(my_pos[0]), int(my_pos[1]))
        tgt = (int(target_cell[0]), int(target_cell[1]))
        # Check if anyone is predicted to traverse the reverse edge at t=1
        reverse_edge = (tgt[0], tgt[1], my_p[0], my_p[1], 1)
        users = self._edge_res.get(reverse_edge, [])
        others = [h for h in users if h != my_handle]
        return 1.0 if others else 0.0

    def lane_agent_count(
        self, corridor_cells: List[Tuple[int, int]], my_handle: int
    ) -> int:
        """Count how many other agents are currently positioned in any corridor cell.
        
        Uses CURRENT positions (not predicted) — represents immediate density.
        """
        if not corridor_cells:
            return 0
        cell_set = set(corridor_cells)
        count = 0
        for agent in self.env.agents:
            if agent.handle == my_handle:
                continue
            if agent.position is None:
                continue
            if (int(agent.position[0]), int(agent.position[1])) in cell_set:
                count += 1
        return count

    # ─────────────────────────────────────────────────────────────────────
    # Global conflict queries
    # ─────────────────────────────────────────────────────────────────────

    def global_conflict_pressure(self) -> float:
        """Fraction of vertex-cells with multi-agent reservations, normalized."""
        n_active = max(1, self._n_active_agents)
        # Normalise: each active pair can contribute up to ~horizon conflicts
        max_possible = n_active * self.horizon
        return float(np.clip(self._n_predicted_conflicts / max(1, max_possible), 0.0, 1.0))

    def my_priority_global(self, my_handle: int) -> float:
        """Rank by remaining shortest-path distance."""
        my_d = self._sp_distance_at_start.get(my_handle, float("inf"))
                
        if not np.isfinite(my_d):
            return 0.0
        finite_distances = [
            d for d in self._sp_distance_at_start.values() if np.isfinite(d)
        ]
        if len(finite_distances) <= 1:
            return 1.0
        sorted_dists = sorted(finite_distances)
        rank_idx = sorted_dists.index(my_d)
        return float(1.0 - rank_idx / max(1, len(sorted_dists) - 1))


    def my_sp_blocked_score(self, my_handle: int) -> float:
        """How much of my predicted path is occupied by other agents' predictions?
        
        Counts vertex-conflicts along my own path (excluding myself).
        Returns [0, 1]: 1.0 = every step blocked, 0.0 = clear path.
        """
        my_path = self._predicted_paths.get(my_handle, [])
        if len(my_path) <= 1:
            return 0.0

        n_blocked = 0
        # Skip t=0 (current position is mine alone by definition)
        for t in range(1, len(my_path)):
            pos = my_path[t]
            users = self._vertex_res.get((pos[0], pos[1], t), [])
            others = [h for h in users if h != my_handle]
            if others:
                n_blocked += 1

        return float(n_blocked / max(1, len(my_path) - 1))

    def expected_yield_count(self, my_handle: int) -> float:
        """If I stop now, how many agents benefit?
        
        An agent A "benefits from my yielding" if:
          - A's predicted path includes a cell that's also in my predicted path
          - A has higher priority (smaller SP distance) than me
        
        Returns [0, 1]: fraction of all active agents that would benefit.
        """
        my_path = self._predicted_paths.get(my_handle, [])
        if len(my_path) <= 1:
            return 0.0
        # Build set over (r, c) only — direction-agnostic for overlap detection.
        # (Path entries are now (r, c, dir) tuples; we strip dir here.)
        my_path_set = {(p[0], p[1]) for p in my_path}

        my_d = self._sp_distance_at_start.get(my_handle, float("inf"))
        if not np.isfinite(my_d):
            return 0.0

        n_benefit = 0
        n_others = 0
        for handle, path in self._predicted_paths.items():
            if handle == my_handle:
                continue
            if len(path) <= 1:
                continue
            n_others += 1

            # Higher priority = shorter remaining distance
            other_d = self._sp_distance_at_start.get(handle, float("inf"))
            if not np.isfinite(other_d):
                continue
            if other_d >= my_d:
                continue  # not higher priority → my yield doesn't help them strategically

            # Check overlap
            other_path_set = {(p[0], p[1]) for p in path}
            if my_path_set & other_path_set:
                n_benefit += 1

        if n_others == 0:
            return 0.0
        return float(n_benefit / n_others)

