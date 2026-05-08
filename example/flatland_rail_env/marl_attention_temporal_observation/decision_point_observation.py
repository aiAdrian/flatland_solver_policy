"""
DecisionPointObservation
========================

Observation builder focused on Flatland's decision points on the directed rail
graph (nodes = (row, col, dir), edges = legal transitions).

Feature layout (length = 54, all values in [0, 1]):

    [ 0]      is_switch  — 1.0 if agent is at a diverging switch, else 0
    [ 1- 3]   shortest_path_hint (one-hot left/fwd/right)
    [ 4]      is_merge   — 1.0 if agent is approaching a merge, else 0
    [ 5]      local_deadlock — 1.0 if confirmed head-on deadlock detected ahead
    [ 6-13]   switch left  block (progress_gain, deadlock_signal, switches_norm,
                                  branch_dist_norm, target_found, abort,
                                  deadlock_ahead_binary, padding)
    [14-21]   switch forward block (same layout)
    [22-29]   switch right  block (same layout)
    [30]      padding
    [31-35]   merge forward sub-block (deadlock_signal, switches_norm,
                                       target_found, abort, deadlock_ahead_binary)
    [36-40]   merge backward sub-block (same layout)
    [41-47]   one-hot of agent.state.value (TrainState 0..6)
    [48-52]   one-hot of last saved action (DO_NOTHING/L/F/R/STOP)
    [53]      priority_rank — normalised rank by remaining path distance in [0,1]

Fixes vs. old layout:
  - [0] was scalar dt/8 (bitfield as float) → now binary is_switch flag.
  - [4] was unused padding → now binary is_merge flag.
  - [5] was raw _detect_deadlock() return (-1/0/1..16) → now binary 0/1.
  - [6] was priority_rank (immediately overwritten by switch-left progress_gain
        when decision_type & 2) → rank moved to [53].
  - [53] was explicit duplicate of [5] → now priority_rank.
  - Branch deadlock_ahead slots [12/20/28/35/40]: raw _detect_deadlock() output
        → now binary via _encode_detect_deadlock().
NOTE: Coordination signals (wait_intent, go_intent, etc.) are computed in the
Policy Network using LSTM, not in the observation.
"""

from typing import List

import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_count_nonzero, fast_argmax


_UNREACHABLE = -1.0


class DecisionPointObservation(ObservationBuilder):
    OBS_SIZE = 54

    def __init__(self):
        super().__init__()
        self.env = None
        self.feature_len = DecisionPointObservation.OBS_SIZE
        self.agent_map = None
        self._max_dist = 1.0
        if not getattr(type(self), "_banner_printed", False):
            print(">> DecisionPointObservation loaded.")
            type(self)._banner_printed = True

    def set_env(self, env):
        self.env = env

    def reset(self):
        self.agent_map = np.zeros((self.env.height, self.env.width), dtype=np.int32) - 1

    @staticmethod
    def getObservationSize() -> int:
        return DecisionPointObservation.OBS_SIZE

    @staticmethod
    def _encode_detect_deadlock(raw: float) -> float:
        """Normalise _detect_deadlock() raw return value to binary [0, 1].

        _detect_deadlock returns:
          -1  →  safe / no corridor ahead (treat as no deadlock)
           0  →  unknown / timeout       (treat as no deadlock)
          >0  →  step-distance to confirmed head-on deadlock → 1.0
        """
        return 1.0 if raw > 0 else 0.0

    @staticmethod
    def _encode_deadlock_signal(deadlock_flag: float) -> float:
        if deadlock_flag is None:
            return 0.0
        if deadlock_flag >= 1.0:
            return 1.0 if deadlock_flag >= 1.5 else 0.85
        return 0.0

    def get(self, handle: int = 0):
        features = np.zeros(self.feature_len, dtype=np.float32)

        agent = self.env.agents[handle]
        pos = agent.position if agent.position is not None else agent.initial_position
        direction = agent.direction if agent.direction is not None else agent.initial_direction
        target = agent.target

        if pos is None or target is None:
            # should never happen, bust just in case to ensure no crashes
            return (features, [])

        distance_map = self.env.distance_map.get()
        curr_dist_raw = distance_map[handle, pos[0], pos[1], direction]
        if curr_dist_raw == np.inf:
            return (features-1, [])

        max_dist = self._max_dist
        curr_dist_norm = float(curr_dist_raw) / max_dist
        transitions = self.env.rail.get_transitions(*pos, direction)

        merge_switch = False
        for rel_dir in (-1, 0, 1):
            ndir = (direction + rel_dir) % 4
            if not transitions[ndir]:
                continue
            next_pos = get_new_position(pos, ndir)
            ntransitions = self.env.rail.get_transitions(*next_pos, ndir)
            if fast_count_nonzero(ntransitions) == 1:
                for d in range(4):
                    next_transitions = self.env.rail.get_transitions(*next_pos, d)
                    if fast_count_nonzero(next_transitions) > 1:
                        merge_switch = True
                        break
            if merge_switch:
                break

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

        # [0] is_switch binary (was: scalar dt/8 — a bitfield as float, unlearnable)
        features[0] = 1.0 if (decision_type & 2) else 0.0
        features[1:4] = self._shortest_path_action_hint(handle, pos, direction, transitions, distance_map)
        # [4] is_merge binary (was: unused padding)
        features[4] = 1.0 if (decision_type & 4) else 0.0
        # [5] local_deadlock binary (was: raw _detect_deadlock return, range -1..16)
        features[5] = self._encode_detect_deadlock(self._detect_deadlock(handle, pos, direction))

        all_distance = []
        for idx, a in enumerate(self.env.agents):
            apos = a.position if a.position is not None else a.initial_position
            adir = a.direction if a.direction is not None else a.initial_direction
            if apos is None or adir is None:
                adist = np.inf
            else:
                adist = float(distance_map[a.handle, apos[0], apos[1], adir])
            all_distance.append((a.handle, adist, idx))

        # Stable sort by value. For ties, first occurrence in the original list wins.
        all_distance.sort(key=lambda x: (x[1], x[2]))
        value_to_rank = {}
        next_rank = 1
        handle_to_rank = {}
        for h, dist, _ in all_distance:
            if dist not in value_to_rank:
                value_to_rank[dist] = next_rank
                next_rank += 1
            handle_to_rank[h] = value_to_rank[dist]

        # priority_rank moved to [53] — feature [6] is the start of the
        # switch-left block and was silently overwritten whenever decision_type & 2.
        priority_rank = float(handle_to_rank.get(handle, next_rank)) / next_rank

        opp_agents = set()
        visited_type_2: set = set()
        visited_type_3_fwd: set = set()
        visited_type_3_bwd: set = set()

        if decision_type & 2:
            for rel_dir in (-1, 0, 1):
                abs_dir = (direction + rel_dir) % 4
                base = 6 + (rel_dir + 1) * 8
                if transitions[abs_dir]:
                    npos = get_new_position(pos, abs_dir)
                    branch_dist, deadlock, switches, seen, abort, target_found, visited_type_2 = \
                        self._navigate_direction(handle, npos, abs_dir, target, False)
                    opp_agents.update(seen)
                    branch_dist_norm = self._normalise_distance(branch_dist, max_dist)
                    switches_norm = self._normalise_count(switches)
                    progress_gain = float(np.clip(curr_dist_norm - branch_dist_norm, 0.0, 1.0))
                    features[base + 0] = progress_gain
                    features[base + 1] = self._encode_deadlock_signal(deadlock)
                    features[base + 2] = switches_norm
                    features[base + 3] = branch_dist_norm
                    features[base + 4] = target_found
                    features[base + 5] = abort
                    # deadlock_ahead: normalized binary (was: raw _detect_deadlock range -1..16)
                    features[base + 6] = self._encode_detect_deadlock(
                        self._detect_deadlock(handle, npos, abs_dir))
                    features[base + 7] = 0.0
                else:
                    features[base + 0] = 0.0
                    features[base + 1] = 0.0
                    features[base + 2] = 0.0
                    features[base + 3] = 0.0
                    features[base + 4] = 0.0
                    features[base + 5] = 1.0
                    features[base + 6] = 0.0
                    features[base + 7] = 0.0

        if decision_type & 4:
            forward_dir = fast_argmax(transitions)
            npos_fwd = get_new_position(pos, forward_dir)

            _, deadlock_fwd, switches_fwd, seen_fwd, abort_fwd, target_found_fwd, visited_type_3_fwd = \
                self._navigate_direction(handle, npos_fwd, forward_dir, target, False)
            opp_agents.update(seen_fwd)
            features[31] = self._encode_deadlock_signal(deadlock_fwd)
            features[32] = self._normalise_count(switches_fwd)
            features[33] = target_found_fwd
            features[34] = abort_fwd
            features[35] = self._encode_detect_deadlock(
                self._detect_deadlock(handle, npos_fwd, forward_dir))

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
                features[36] = self._encode_deadlock_signal(deadlock_bwd)
                features[37] = self._normalise_count(switches_bwd)
                features[38] = target_found_bwd
                features[39] = abort_bwd
                features[40] = self._encode_detect_deadlock(
                    self._detect_deadlock(handle, bwd_pos, bwd_dir))

        # State one-hot: features[41-47]
        state_value = int(agent.state.value)
        if 0 <= state_value <= 6:
            features[41 + state_value] = 1.0

        # Action one-hot: features[48-52]
        if agent.action_saver.is_action_saved:
            sa = int(agent.action_saver.saved_action)
            if 0 <= sa <= 4:
                features[48 + sa] = 1.0

        # [53] priority_rank (was: explicit duplicate of [5] — wasted dimension)
        features[53] = priority_rank

        # NOTE: Coordination signals are computed in Policy Network via LSTM
        # Do NOT compute them here in observation

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
        for handle in handles:
            obs_self, obs_others = self.get(handle)
            result.append((obs_self, obs_others))

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
        best_hint = [0.0, 0.0, 0.0]
        min_dist = np.inf
        best_idx = 1
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
                        return cur_dist, 1, num_switches, 0, 0
                elif agent_idx == handle and agent_idx != -1:
                    return cur_dist, 2, num_switches, 0, 0

            num_trans = fast_count_nonzero(transitions)
            if num_trans > 1:
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
        """Test if agent will have 100% certain deadlock by following best path until next switch.
        Returns: step distance to confirmed head-on deadlock, or 0 if safe/unclear."""
        max_steps = 16
        s = 0
        
        while s < max_steps:
            s += 1
            transitions = self.env.rail.get_transitions(*pos, direction)
            ndir = fast_argmax(transitions)
            
            # Can we continue in best direction?
            if not transitions[ndir]:
                return -1
            
            # Check if we've reached a switch
            num_trans = fast_count_nonzero(transitions)
            if num_trans > 1:
                # Reached a switch/junction - stop here, not a 100% deadlock
                return -1
            
            npos = get_new_position(pos, ndir)
            
            # Check for agent in next position
            agent_idx = self.agent_map[npos]
            if agent_idx != -1 and agent_idx != handle:
                other = self.env.agents[agent_idx]
                # 100% deadlock ONLY if head-on collision (opposite direction)
                if self._is_opposite_direction(ndir, other.direction):
                    return s
            
            pos = npos
            direction = ndir
        
        return 0

    def _estimate_ttc_conflict_risk(self, handle, pos, direction, horizon: int = 6) -> float:
        """Estimate short-horizon conflict risk as 1 - normalized time-to-conflict."""
        transitions = self.env.rail.get_transitions(*pos, direction)
        min_steps = horizon + 1

        for rel in (-1, 0, 1):
            ndir = (direction + rel) % 4
            if not transitions[ndir]:
                continue

            cur_pos = get_new_position(pos, ndir)
            cur_dir = ndir

            for step in range(1, horizon + 1):
                if cur_pos[0] < 0 or cur_pos[0] >= self.env.height or cur_pos[1] < 0 or cur_pos[1] >= self.env.width:
                    break

                other_handle = self.agent_map[cur_pos] if self.agent_map is not None else -1
                if other_handle != -1 and other_handle != handle:
                    other = self.env.agents[other_handle]
                    if self._is_opposite_direction(cur_dir, other.direction) or self._is_forward_only(cur_pos, cur_dir, cur_pos, other.direction):
                        min_steps = min(min_steps, step)
                        break

                next_trans = self.env.rail.get_transitions(*cur_pos, cur_dir)
                if fast_count_nonzero(next_trans) == 0:
                    break
                if fast_count_nonzero(next_trans) > 1:
                    # At branch points uncertainty is high enough; stop rollout here.
                    break
                cur_dir = fast_argmax(next_trans)
                cur_pos = get_new_position(cur_pos, cur_dir)

        if min_steps > horizon:
            return 0.0
        return float(np.clip(1.0 - (float(min_steps) - 1.0) / float(horizon), 0.0, 1.0))

    def _estimate_right_of_way(self, handle: int, curr_dist_norm: float, opp_agents: set) -> float:
        """Estimate relative priority against currently relevant opponents."""
        my_priority = float(np.clip(1.0 - curr_dist_norm, 0.0, 1.0))
        if len(opp_agents) == 0:
            return my_priority

        distance_map = self.env.distance_map.get()
        opp_prios = []
        for opp_h in opp_agents:
            if opp_h < 0 or opp_h >= len(self.env.agents):
                continue
            opp = self.env.agents[opp_h]
            opp_pos = opp.position if opp.position is not None else opp.initial_position
            opp_dir = opp.direction if opp.direction is not None else opp.initial_direction
            if opp_pos is None or opp_dir is None:
                continue
            d = distance_map[opp_h, opp_pos[0], opp_pos[1], opp_dir]
            if d == np.inf or self._max_dist <= 0:
                continue
            opp_prios.append(float(np.clip(1.0 - (float(d) / self._max_dist), 0.0, 1.0)))

        if len(opp_prios) == 0:
            return my_priority

        opp_priority = max(opp_prios)
        return float(np.clip(0.5 + 0.5 * (my_priority - opp_priority), 0.0, 1.0))

    def _estimate_cycle_risk(self, handle: int, pos, direction, lookahead: int = 3) -> float:
        """Local cycle risk proxy using repeated mutual-block patterns in a short horizon."""
        frontier = [(pos, direction, 0)]
        visited = set()
        conflict_hits = 0

        while frontier:
            cur_pos, cur_dir, depth = frontier.pop()
            if depth >= lookahead:
                continue
            key = (cur_pos, cur_dir, depth)
            if key in visited:
                continue
            visited.add(key)

            trans = self.env.rail.get_transitions(*cur_pos, cur_dir)
            for ndir in range(4):
                if not trans[ndir]:
                    continue
                npos = get_new_position(cur_pos, ndir)
                if npos[0] < 0 or npos[0] >= self.env.height or npos[1] < 0 or npos[1] >= self.env.width:
                    continue
                other_handle = self.agent_map[npos] if self.agent_map is not None else -1
                if other_handle != -1 and other_handle != handle:
                    other = self.env.agents[other_handle]
                    if self._is_opposite_direction(ndir, other.direction):
                        conflict_hits += 1
                frontier.append((npos, ndir, depth + 1))

        return float(np.clip(conflict_hits / 3.0, 0.0, 1.0))

    @staticmethod
    def _is_opposite_direction(dir1, dir2) -> bool:
        return (dir1 + 2) % 4 == dir2

    def _is_forward_only(self, pos1, dir1, pos2, dir2) -> bool:
        t1 = self.env.rail.get_transitions(*pos1, dir1)
        t2 = self.env.rail.get_transitions(*pos2, dir2)
        return fast_count_nonzero(t1) == 1 and fast_count_nonzero(t2) == 1
