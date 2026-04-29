"""
DecisionPointObservation
========================

Observation builder focused on Flatland's decision points on the directed rail
graph (nodes = (row, col, dir), edges = legal transitions):

    decision_type bitfield
        1 = READY_TO_DEPART (start decision)
        2 = at a switch (routing: branch left / forward / right)
        4 = one cell before a merge/crossing switch (ordering / let-pass)
        8 = DONE

References
----------
Mohanty et al. (2020) "Flatland-RL: Multi-Agent RL on Trains",
    arXiv:2012.05893 -- defines the directed-graph train scheduling problem,
    grid encoding and `distance_map` (per-agent BFS distance from each
    (row, col, dir) to the agent's target).
Laurent et al. (2021) "Flatland Competition 2020: MAPF and MARL ...",
    arXiv:2103.16511 -- decision-point-aware observations are used by the
    top scoring solutions; normalised features in [0, 1] are crucial for the
    NN to converge (LeCun et al. 1998 "Efficient BackProp", Sec. 4.3).

Feature layout (length = 43, all values normalised to roughly [0, 1]):

    [ 0]      decision_type
    [ 1- 3]   shortest_path_hint  (one-hot left/fwd/right of distance map)
    [ 4- 9]   switch left   block   (curr_dist_norm, deadlock, switches_norm,
                                     branch_dist_norm, target_found, abort)
    [10-15]   switch forward block  (same layout)
    [16-21]   switch right  block   (same layout)
    [22-25]   merge forward sub-block (deadlock, switches_norm,
                                       target_found, abort)
    [26-29]   merge backward sub-block (same layout)
    [30-36]   one-hot of agent.state.value (TrainState 0..6)
    [37-41]   one-hot of last saved action (DO_NOTHING/L/F/R/STOP)
    [42]      local_deadlock flag

For unreachable branches the per-branch fields collapse to 0 except for an
explicit mask delivered through `target_found` / `abort`. All raw distance
values are divided by the maximum *reachable* distance found in the per-agent
distance map, so the network only ever sees inputs in [0, 1].

Important: Flatland is a *directed graph* problem on a track network. The grid
size (height * width) is NOT a meaningful normalisation constant -- we use the
actual max reachable distance from the BFS-based distance map.
"""

from typing import List, Tuple

import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_count_nonzero, fast_argmax


# Sentinel value for unreachable branches *before* normalisation.
_UNREACHABLE = -1.0


class DecisionPointObservation(ObservationBuilder):
    OBS_SIZE = 43

    def __init__(self):
        super().__init__()
        self.env = None
        self.feature_len = DecisionPointObservation.OBS_SIZE
        self.agent_map = None
        # cached per get_many call
        self._max_dist = 1.0
        print(">> DecisionPointObservation loaded.")

    # ------------------------------------------------------------------ API

    def set_env(self, env):
        self.env = env

    def reset(self):
        self.agent_map = np.zeros((self.env.height, self.env.width), dtype=np.int32) - 1

    @staticmethod
    def getObservationSize() -> int:
        return DecisionPointObservation.OBS_SIZE

    # ----------------------------------------------------------------- core

    def get(self, handle: int = 0):
        features = np.zeros(self.feature_len, dtype=np.float32)

        agent = self.env.agents[handle]
        pos = agent.position if agent.position is not None else agent.initial_position
        direction = agent.direction if agent.direction is not None else agent.initial_direction
        target = agent.target

        if pos is None or target is None:
            # not on map yet, no target -> return all-zero observation (neutral input)
            return (features, [])

        distance_map = self.env.distance_map.get()
        curr_dist_raw = distance_map[handle, pos[0], pos[1], direction]
        if curr_dist_raw == np.inf:
            return (features, [])

        max_dist = self._max_dist
        curr_dist_norm = float(curr_dist_raw) / max_dist

        transitions = self.env.rail.get_transitions(*pos, direction)

        # Detect "merge / crossing" condition: is the next cell a switch which
        # the agent ENTERS in a non-branching direction?
        merge_switch = False
        for rel_dir in (-1, 0, 1):
            ndir = (direction + rel_dir) % 4
            if transitions[ndir]:
                next_pos = get_new_position(pos, ndir)
                for d in range(4):
                    next_transitions = self.env.rail.get_transitions(*next_pos, d)
                    if fast_count_nonzero(next_transitions) > 1 and d != ndir:
                        merge_switch = True
                        break
                if merge_switch:
                    break

        # ---- decision_type classification -----------------------------------
        decision_type = 0
        if agent.state.name == "READY_TO_DEPART":
            decision_type = 1
        else:
            if fast_count_nonzero(transitions) > 1:
                decision_type += 2
            if merge_switch:
                decision_type += 4
        if agent.state.name == "DONE":
            decision_type = 8

        features[0] = decision_type
        features[1:4] = self._shortest_path_action_hint(handle, pos, direction, transitions, distance_map)
        features[42] = self._detect_deadlock(handle, pos, direction)

        opp_agents = set()
        visited_type_2: set = set()
        visited_type_3_fwd: set = set()
        visited_type_3_bwd: set = set()

        # ---- decision_type & 2 : at a switch (branch decision) --------------
        if decision_type & 2:
            for rel_dir in (-1, 0, 1):
                abs_dir = (direction + rel_dir) % 4
                base = 4 + (rel_dir + 1) * 6
                if transitions[abs_dir]:
                    npos = get_new_position(pos, abs_dir)
                    branch_dist, deadlock, switches, seen, abort, target_found, visited_type_2 = \
                        self._navigate_direction(handle, npos, abs_dir, target, False)
                    opp_agents.update(seen)
                    branch_dist_norm = self._normalise_distance(branch_dist, max_dist)
                    switches_norm = self._normalise_count(switches)
                    features[base + 0] = curr_dist_norm
                    features[base + 1] = deadlock
                    features[base + 2] = switches_norm
                    features[base + 3] = branch_dist_norm
                    features[base + 4] = target_found
                    features[base + 5] = abort
                else:
                    # Unreachable branch -> all zero except an explicit "abort"
                    # signal so the network can mask it out.
                    features[base + 0] = curr_dist_norm
                    features[base + 1] = 0.0
                    features[base + 2] = 0.0
                    features[base + 3] = 0.0
                    features[base + 4] = 0.0
                    features[base + 5] = 1.0  # abort = unreachable

        # ---- decision_type & 4 : one cell before a merge / crossing ---------
        if decision_type & 4:
            forward_dir = int(np.argmax(transitions))
            npos_fwd = get_new_position(pos, forward_dir)

            _, deadlock_fwd, switches_fwd, seen_fwd, abort_fwd, target_found_fwd, visited_type_3_fwd = \
                self._navigate_direction(handle, npos_fwd, forward_dir, target, False)
            opp_agents.update(seen_fwd)
            features[22] = deadlock_fwd
            features[23] = self._normalise_count(switches_fwd)
            features[24] = target_found_fwd
            features[25] = abort_fwd

            bwd_pos = None
            bwd_dir = None
            for d in range(1, 4):
                nd = (forward_dir + d) % 4
                nt = self.env.rail.get_transitions(*npos_fwd, nd)
                if fast_count_nonzero(nt) > 1:
                    for i in range(4):
                        if nt[i]:
                            tmp_pos = get_new_position(npos_fwd, i)
                            if tmp_pos != pos:
                                bwd_pos = tmp_pos
                                bwd_dir = i
                                break
                if bwd_pos is not None:
                    break

            if bwd_pos is not None:
                _, deadlock_bwd, switches_bwd, seen_bwd, abort_bwd, target_found_bwd, visited_type_3_bwd = \
                    self._navigate_direction(handle, bwd_pos, bwd_dir, target, True)
                opp_agents.update(seen_bwd)
                features[26] = deadlock_bwd
                features[27] = self._normalise_count(switches_bwd)
                features[28] = target_found_bwd
                features[29] = abort_bwd

        # ---- one-hot agent state and last saved action ----------------------
        # TrainState values: WAITING=0, READY_TO_DEPART=1, MALFUNCTION_OFF_MAP=2,
        # MOVING=3, STOPPED=4, MALFUNCTION=5, DONE=6
        state_value = int(agent.state.value)
        if 0 <= state_value <= 6:
            features[30 + state_value] = 1.0

        if agent.action_saver.is_action_saved:
            sa = int(agent.action_saver.saved_action)
            if 0 <= sa <= 4:
                features[37 + sa] = 1.0

        # ---- bookkeeping for the renderer -----------------------------------
        visited: List = []
        for a in visited_type_2:
            visited.append(a[0])
        for a in visited_type_3_fwd:
            visited.append(a[0])
        for a in visited_type_3_bwd:
            visited.append(a[0])
        self.env.dev_obs_dict.update({handle: visited})

        agent.cur_opp_agent_handles = list(opp_agents)
        return (features, agent.cur_opp_agent_handles)

    def get_many(self, handles: list = None):
        if handles is None:
            handles = list(range(len(self.env.agents)))

        # rebuild the agent occupancy map
        self.agent_map = np.zeros((self.env.height, self.env.width), dtype=np.int32) - 1
        for agent in self.env.agents:
            if agent.position is not None:
                self.agent_map[agent.position] = agent.handle

        # cache the maximum reachable distance for normalisation (per call)
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
        for handle in handles:
            obs_self, obs_others = self.get(handle)
            result.append((obs_self, obs_others))

        for agent in self.env.agents:
            agent.opp_agent_handles = agent.cur_opp_agent_handles

        return result

    # ----------------------------------------------------------- helpers ---

    @staticmethod
    def _normalise_distance(value: float, max_dist: float) -> float:
        """Map a raw distance value to [0, 1]. Unreachable / sentinel -> 0."""
        if value is None or value == _UNREACHABLE:
            return 0.0
        if not np.isfinite(value):
            return 0.0
        if max_dist <= 0:
            return 0.0
        return float(np.clip(value / max_dist, 0.0, 1.0))

    @staticmethod
    def _normalise_count(value: float) -> float:
        """Soft-normalise a non-negative count into [0, 1] via x / (x + 8)."""
        if value is None or value < 0:
            return 0.0
        return float(value) / (float(value) + 8.0)

    def _shortest_path_action_hint(self, handle, pos, direction, transitions, distance_map):
        best_hint = [0.0, 0.0, 0.0]  # [left, forward, right]
        min_dist = np.inf
        best_idx = 1  # default: forward
        for idx, rel in enumerate((-1, 0, 1)):
            ndir = (direction + rel) % 4
            if transitions[ndir]:
                npos = get_new_position(pos, ndir)
                dist = distance_map[handle, npos[0], npos[1], ndir]
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
        best_hint[best_idx] = 1.0
        return best_hint

    def _navigate_direction(self, handle, start_pos, start_dir, target, backward_trace, max_steps: int = 100):
        """
        Bounded DFS on the directed rail graph from `start_pos`/`start_dir`.

        Returns (dist, deadlock_flag, num_switches, seen_agents, abort_flag,
                 target_found_flag, visited_set).

        Uses a per-call mutable controller for cycle detection and step budget,
        so a single call respects `max_steps` regardless of branching factor.
        """
        env = self.env
        distance_map = env.distance_map.get()

        controller = {
            'count': 0,
            'visited': set(),
            'seen_agents': set(),
        }

        max_depth = max(64, int(4 * max_steps))

        def dfs(pos, dirn, num_switches, backward, budget, depth):
            cur_dist = distance_map[handle, pos[0], pos[1], dirn]

            if controller['count'] >= budget:
                return cur_dist, 0, num_switches, 1, -1
            if depth >= max_depth:
                # Hard stop against pathological recursion in cyclic rail topologies.
                return cur_dist, -1, num_switches, 1, -1
            if pos == target and not backward:
                return cur_dist, 0, num_switches, 0, 1
            if (pos, dirn) in controller['visited']:
                return cur_dist, -1, num_switches, 1, -1
            if cur_dist == np.inf and not backward:
                return _UNREACHABLE, -1, num_switches, 1, -1

            controller['visited'].add((pos, dirn))
            controller['count'] += 1

            transitions = env.rail.get_transitions(*pos, dirn)

            if self.agent_map is not None:
                agent_idx = self.agent_map[pos]
                if agent_idx != -1 and agent_idx != handle:
                    other_dir = env.agents[agent_idx].direction
                    if other_dir != dirn and not backward:
                        controller['seen_agents'].add(agent_idx)
                    if other_dir != dirn:
                        # Head-on with another agent on the forward search.
                        return cur_dist, 1, num_switches, 0, 0
                elif agent_idx == handle and agent_idx != -1:
                    # We met ourselves on the backward trace.
                    return cur_dist, 2, num_switches, 0, 0

            num_trans = fast_count_nonzero(transitions)
            if num_trans > 1:
                # at a switch -> rank alternatives by distance map
                alternatives = []
                for ndir in range(4):
                    if transitions[ndir]:
                        npos = get_new_position(pos, ndir)
                        if (npos, ndir) not in controller['visited']:
                            d_alt = distance_map[handle, npos[0], npos[1], ndir]
                            alternatives.append((d_alt, ndir, npos))
                alternatives.sort(key=lambda x: x[0])

                if not backward:
                    increment = 0
                    for _, ndir, npos in alternatives:
                        rd, rdl, rns, rab, rtf = dfs(npos, ndir, num_switches + increment, backward, budget, depth + 1)
                        if rdl < 1:
                            return max(rd, cur_dist), rdl, rns, rab, rtf
                        increment = 1
                else:
                    if not alternatives:
                        return cur_dist, 1, num_switches, 0, 0
                    sums = [0.0, 0.0, 0.0, 0.0, 0.0]
                    for _, ndir, npos in alternatives:
                        rd, rdl, rns, rab, rtf = dfs(npos, ndir, num_switches + 1, backward, budget, depth + 1)
                        sums[0] += rd
                        sums[1] += rdl
                        sums[2] += rns
                        sums[3] += rab
                        sums[4] += rtf
                    n = float(len(alternatives))
                    return (max(sums[0] / n, cur_dist), sums[1] / n, sums[2] / n, sums[3] / n, sums[4] / n)
            else:
                # corridor cell -> follow the only legal transition
                ndir = fast_argmax(transitions)
                npos = get_new_position(pos, ndir)
                if (npos, ndir) not in controller['visited']:
                    rd, rdl, rns, rab, rtf = dfs(npos, ndir, num_switches, backward, budget, depth + 1)
                    return max(rd, cur_dist), rdl, rns, rab, rtf

            return cur_dist, 1, num_switches, 0, 0

        dist, deadlock_flag, num_switches, abort_flag, target_found_flag = \
            dfs(start_pos, start_dir, 0, backward_trace, max_steps, 0)
        seen_agents = sorted(controller['seen_agents'])
        return dist, deadlock_flag, num_switches, seen_agents, abort_flag, target_found_flag, controller['visited']

    def _detect_deadlock(self, handle, pos, direction):
        """Local "nose-to-nose, both forward-only" deadlock check (1/0)."""
        transitions = self.env.rail.get_transitions(*pos, direction)
        for ndir in range(4):
            if not transitions[ndir]:
                continue
            npos = get_new_position(pos, ndir)
            agent_idx = self.agent_map[npos]
            if agent_idx == -1 or agent_idx == handle:
                continue
            other = self.env.agents[agent_idx]
            if not self._is_opposite_direction(direction, other.direction):
                continue
            if self._is_forward_only(pos, direction, npos, other.direction):
                return 1
        return 0

    @staticmethod
    def _is_opposite_direction(dir1, dir2) -> bool:
        return (dir1 + 2) % 4 == dir2

    def _is_forward_only(self, pos1, dir1, pos2, dir2) -> bool:
        t1 = self.env.rail.get_transitions(*pos1, dir1)
        t2 = self.env.rail.get_transitions(*pos2, dir2)
        return fast_count_nonzero(t1) == 1 and fast_count_nonzero(t2) == 1
