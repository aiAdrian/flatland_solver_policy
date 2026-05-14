"""
DecisionPointObservation
=========================

Diese Klasse implementiert einen Beobachtungs-Builder für Flatland, der sich auf Entscheidungspunkte im gerichteten Schienennetz konzentriert. 
Die Beobachtungen umfassen Informationen über Schalter, Deadlocks, kürzeste Pfade und andere relevante Merkmale, die für Multi-Agenten-Verstärkungslernen nützlich sind.

Hauptmerkmale:
- **is_switch**: Gibt an, ob sich der Agent an einem Schalter befindet.
- **shortest_path_hint**: Ein One-Hot-Vektor, der die Richtung des kürzesten Pfades angibt (links, vorwärts, rechts).
- **local_deadlock**: Binary-Wert, der Deadlock-Risiken anzeigt.
- **TreeLSTM-Integration**: Verarbeitet Baumdaten aus der lokalen Suche, um hierarchische Beobachtungen zu erstellen.

Die Klasse ist unabhängig von Solver- oder Reward-Shaping-Mechanismen und kann direkt in Multi-Agenten-Umgebungen verwendet werden.

Feature-Layout (Länge = 66, Wertebereich [0, 1]):
- [0]: is_switch (1.0, wenn Schalter, sonst 0)
- [1-3]: shortest_path_hint (One-Hot für links/vorwärts/rechts)
- [4]: is_merge (1.0, wenn Merge-Bereich voraus, sonst 0)
- [5]: local_deadlock (1.0, wenn Deadlock-Risiko erkannt, sonst 0)
- [6-13]: Schalter-Features für links (z. B. Fortschritt, Deadlock-Signal, Distanz)
- [14-21]: Schalter-Features für vorwärts
- [22-29]: Schalter-Features für rechts
- [30]: decision_required (1.0, wenn Entscheidung erforderlich, sonst 0)
- [31-35]: Merge-Features vorwärts
- [36-40]: Merge-Features rückwärts
- [41-47]: One-Hot für den Zustand des Agenten (TrainState 0..6)
- [48-52]: One-Hot für die letzte gespeicherte Aktion
- [53]: priority_rank (normalisierte Priorität basierend auf verbleibender Distanz)
- [54-58]: Zelltyp-One-Hot (z. B. OUTSIDE, SWITCH)
- [59-63]: Transition-Features (z. B. FWD->SWI)
- [64]: tree_ctx (TreeLSTM-Kontext aus lokaler Suche, tiefengewichtet)
- [65]: Deadlock-Flag (1.0, wenn Deadlock erkannt, sonst 0)

"""

from typing import List

import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_count_nonzero, fast_argmax
from .decision_point_utils import DecisionPointUtils
from .tree_lstm import TreeLSTM  # Import TreeLSTM module


_UNREACHABLE = -1.0


class DecisionPointObservation(ObservationBuilder):
    """
    Diese Klasse implementiert die Hauptlogik für die Beobachtungen. 
    Sie sammelt Daten über die Umgebung des Agenten, führt eine lokale Suche durch und integriert die Ergebnisse in die Beobachtungsfeatures.

    Methoden:
    - __init__: Initialisiert die Klasse und das TreeLSTM-Modul.
    - set_env: Setzt die Umgebung für die Beobachtungen.
    - reset: Initialisiert die Agentenkarte.
    - _local_search: Führt eine Tiefensuche durch, um Baumdaten zu sammeln.
    - _calculate_deadlock_risk: Berechnet das Deadlock-Risiko entlang eines Pfades.
    - get: Generiert die Beobachtungen für einen bestimmten Agenten.
    - get_many: Generiert Beobachtungen für mehrere Agenten.
    """

    OBS_SIZE = 70
    FEATURE_GROUPS_DOC = [
        ("[0]", "is_switch", "1.0 if agent is on a switch cell (current cell has >1 outgoing transitions)"),
        ("[1-3]", "hint_L/F/R", "shortest-path direction hint one-hot"),
        ("[4]", "is_merge", "1.0 if agent is one step before a true merge node (own cell: one exit; next cell: multiple incoming, single onward)"),
        ("[5]", "local_deadlock", "binary corridor blockage/deadlock risk at current node"),
        ("[6-13]", "swL_*", "left branch metrics (progress, deadlock, distance, target_reachable, abort)"),
        ("[14-21]", "swF_*", "forward branch metrics"),
        ("[22-29]", "swR_*", "right branch metrics"),
        ("[30]", "decision_required", "1.0 if agent must choose now (OUT/MERGING/SWITCH, else 0.0)"),
        ("[31-35]", "mgF_*", "merge-forward metrics"),
        ("[36-40]", "mgB_*", "merge-backward metrics"),
        ("[41-47]", "st_0..st_6", "TrainState one-hot (READY, MOVING, ..., DONE)"),
        ("[48-52]", "act_DN/L/F/R/S", "last saved action one-hot"),
        ("[53]", "priority_rank", "normalized rank by remaining path distance"),
        ("[54-58]", "ct_*", "current cell-type one-hot (OUT, FWD, MRG, SWI, DONE)"),
        ("[59-63]", "tr_*", "5 selected transitions: FWD->FWD/MRG/SWI, SWI->FWD, MRG->FWD"),
        ("[64]", "tree_ctx", "TreeLSTM local-search context (depth-weighted deadlock aggregate)"),
        ("[65]", "deadlock", "1.0 if confirmed corridor deadlock ahead (recursive cycle check)"),
        ("[66-69]", "tree_*", "Raw tree statistics: mean_dl, max_dl, conflict_density, branching_ratio (for learnable tree_proj)"),
    ]

    def __init__(self):
        super().__init__()
        self.env = None
        self.feature_len = DecisionPointObservation.OBS_SIZE
        self.agent_map = None
        self._max_dist = 1.0
        self.search_depth = 5  # Depth limit for local search
        self.tree_lstm = TreeLSTM(input_dim=8, hidden_dim=16)  # Initialize TreeLSTM
        if not getattr(type(self), "_banner_printed", False):
            print(">> DecisionPointObservation geladen.")
            type(self)._banner_printed = True
        if not getattr(type(self), "_feature_layout_printed", False):
            self._print_feature_layout_doc()
            type(self)._feature_layout_printed = True

    def set_env(self, env):
        self.env = env

    def reset(self):
        self.agent_map = np.zeros((self.env.height, self.env.width), dtype=np.int32) - 1

    def _local_search(self, handle, start_pos, start_dir, depth_limit):
        """Robuste, defensive lokale Suche: Sammelt Tree-Features, gibt bei Fehlern Default zurück."""
        try:
            if start_pos is None or start_dir is None or self.env is None or self.env.rail is None:
                print(f"[Warn] _local_search: Ungültige Startdaten für Agent {handle}.")
                return []
            visited = set()
            frontier = [(start_pos, start_dir, 0)]
            tree_data = []
            while frontier:
                current_pos, current_dir, depth = frontier.pop()
                if depth > depth_limit or (current_pos, current_dir) in visited:
                    continue
                visited.add((current_pos, current_dir))
                try:
                    transitions = self.env.rail.get_transitions(*current_pos, current_dir)
                except Exception as e:
                    print(f"[Warn] _local_search: Fehler bei get_transitions: {e}")
                    continue
                num_transitions = fast_count_nonzero(transitions)
                agents_encountered = []
                has_oncoming = False
                if self.agent_map is not None:
                    try:
                        agent_idx = self.agent_map[current_pos]
                        if agent_idx != -1 and agent_idx != handle:
                            agents_encountered.append(agent_idx)
                            other_dir = self.env.agents[agent_idx].direction
                            if other_dir is not None and DecisionPointUtils.is_opposite_direction(current_dir, other_dir):
                                has_oncoming = True
                    except Exception as e:
                        print(f"[Warn] _local_search: Fehler bei agent_map: {e}")
                try:
                    base_risk = self._calculate_deadlock_risk(handle, current_pos, current_dir)
                except Exception as e:
                    print(f"[Warn] _local_search: Fehler bei _calculate_deadlock_risk: {e}")
                    base_risk = 1.0
                adjusted_risk = min(1.0, base_risk + (0.5 if has_oncoming else 0.0))
                node_info = {
                    "pos": current_pos,
                    "dir": current_dir,
                    "depth": depth,
                    "num_transitions": num_transitions,
                    "deadlock_risk": adjusted_risk,
                    "agents_encountered": agents_encountered,
                    "has_oncoming": has_oncoming,
                }
                tree_data.append(node_info)
                for next_dir in range(4):
                    if transitions[next_dir]:
                        next_pos = get_new_position(current_pos, next_dir)
                        frontier.append((next_pos, next_dir, depth + 1))
            return tree_data
        except Exception as e:
            print(f"[Warn] _local_search: Schwerwiegender Fehler: {e}")
            return []

    def _calculate_deadlock_risk(self, handle, pos, direction):
        """Defensive Deadlock-Risk-Berechnung: Gibt bei Fehlern Risiko=1.0 zurück."""
        try:
            if pos is None or direction is None or self.env is None or self.env.rail is None:
                print(f"[Warn] _calculate_deadlock_risk: Ungültige Eingaben für Agent {handle}.")
                return 1.0
            visited = set()
            frontier = [(pos, direction)]
            deadlock_risk = 0.0
            while frontier:
                current_pos, current_dir = frontier.pop()
                if (current_pos, current_dir) in visited:
                    continue
                visited.add((current_pos, current_dir))
                try:
                    transitions = self.env.rail.get_transitions(*current_pos, current_dir)
                except Exception as e:
                    print(f"[Warn] _calculate_deadlock_risk: Fehler bei get_transitions: {e}")
                    deadlock_risk += 1.0
                    continue
                num_transitions = fast_count_nonzero(transitions)
                if num_transitions == 0:
                    deadlock_risk += 1.0
                elif num_transitions > 1:
                    deadlock_risk += 0.5
                for next_dir in range(4):
                    if transitions[next_dir]:
                        next_pos = get_new_position(current_pos, next_dir)
                        frontier.append((next_pos, next_dir))
            return min(deadlock_risk / 10.0, 1.0)
        except Exception as e:
            print(f"[Warn] _calculate_deadlock_risk: Schwerwiegender Fehler: {e}")
            return 1.0

    @staticmethod
    def getObservationSize() -> int:
        return DecisionPointObservation.OBS_SIZE

    @classmethod
    def _print_feature_layout_doc(cls):
        print(">> Beobachtungs-Layout (66D) - kompakte Feature-Übersicht:")
        for idx, name, desc in cls.FEATURE_GROUPS_DOC:
            print(f"   {idx:<8} {name:<14} {desc}")

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
        transitions = self.env.rail.get_transitions(*pos, direction)
        return fast_count_nonzero(transitions) > 1

    def _incoming_degree(self, cell_pos) -> int:
        """Count incoming directed edges to a cell by local 4-neighborhood scan."""
        incoming_edges = set()
        for prev_dir in range(4):
            prev_pos = get_new_position(cell_pos, (prev_dir + 2) % 4)
            if prev_pos[0] < 0 or prev_pos[0] >= self.env.height or prev_pos[1] < 0 or prev_pos[1] >= self.env.width:
                continue
            for d in range(4):
                trans = self.env.rail.get_transitions(*prev_pos, d)
                for nd in range(4):
                    if not trans[nd]:
                        continue
                    npos = get_new_position(prev_pos, nd)
                    if npos == cell_pos:
                        incoming_edges.add((prev_pos[0], prev_pos[1], d, nd))
        return len(incoming_edges)

    def _is_pre_merge_one_exit(self, pos, direction, transitions) -> bool:
        """True if agent is exactly one step before a merge/conflict node with one current exit.

        Semantics for DAG-style routing:
        - current cell: exactly one usable outgoing edge for the current heading
        - next cell: true merge node, i.e. receives multiple incoming edges and
          has a single onward edge for the arriving orientation
        """
        if fast_count_nonzero(transitions) != 1:
            return False
        ndir = fast_argmax(transitions)
        if not transitions[ndir]:
            return False
        # "Nur forward" am aktuellen Knoten: kein Links/Rechts-Entscheid mehr möglich.
        if ndir != direction:
            return False
        next_pos = get_new_position(pos, ndir)
        if next_pos[0] < 0 or next_pos[0] >= self.env.height or next_pos[1] < 0 or next_pos[1] >= self.env.width:
            return False
        in_deg = self._incoming_degree(next_pos)
        if in_deg <= 1:
            return False
        next_transitions_arrival = self.env.rail.get_transitions(*next_pos, ndir)
        return fast_count_nonzero(next_transitions_arrival) == 1

    def _decision_type_at_position(self, pos, direction, target) -> int:
        if pos == target:
            return 8
        transitions = self.env.rail.get_transitions(*pos, direction)
        decision_type = 0
        if self._is_switch_at_current_cell(pos, direction):
            decision_type += 2
        if self._is_pre_merge_one_exit(pos, direction, transitions):
            decision_type += 4
        return decision_type

    def get(self, handle: int = 0):
        raw_features = np.zeros(70, dtype=np.float32)
        try:
            agent = self.env.agents[handle]
            pos = agent.position if agent.position is not None else agent.initial_position
            direction = agent.direction if agent.direction is not None else agent.initial_direction
            target = agent.target
            if pos is None or target is None or direction is None:
                print(f"[Warn] get: Ungültige Agenten-Startdaten für {handle}.")
                return (raw_features, [])
            distance_map = self.env.distance_map.get()
            curr_dist_raw = distance_map[handle, pos[0], pos[1], direction]
            curr_reachable = bool(np.isfinite(curr_dist_raw))
            max_dist = self._max_dist
            # IMPORTANT: np.inf distance is a valid planning state in Flatland.
            # We do not log warnings for it; instead we encode it in features below.
            curr_dist_norm = (float(curr_dist_raw) / max_dist) if curr_reachable else 1.0
            try:
                transitions = self.env.rail.get_transitions(*pos, direction)
            except Exception as e:
                print(f"[Warn] get: Fehler bei get_transitions: {e}")
                transitions = [0, 0, 0, 0]
            # Lokale Suche + TreeLSTM-Aggregation → Slot [64] (reserviert)
            tree_data = self._local_search(handle, pos, direction, self.search_depth)
            local_search_seen_agents = set()
            for node in tree_data:
                for seen_h in node.get("agents_encountered", []):
                    if seen_h != handle:
                        local_search_seen_agents.add(int(seen_h))
            try:
                lstm_output = self._process_tree_data_with_lstm(tree_data)
                raw_features[64] = lstm_output[0]  # tiefengewichtetes Deadlock-Aggregat
            except Exception as e:
                print(f"[Warn] get: Fehler bei TreeLSTM: {e}")
                raw_features[64] = 0.0
            merge_switch = False
            try:
                merge_switch = self._is_pre_merge_one_exit(pos, direction, transitions)
            except Exception as e:
                print(f"[Warn] get: Fehler bei _is_pre_merge_one_exit: {e}")
            decision_type = 0
            if agent.state.name == "READY_TO_DEPART":
                decision_type = 1
            else:
                if self._is_switch_at_current_cell(pos, direction):
                    decision_type += 2
                if merge_switch:
                    decision_type += 4
            if agent.state.name == "DONE":
                decision_type = 8
            raw_features[0] = 1.0 if (decision_type & 2) else 0.0
            try:
                raw_features[1:4] = self._shortest_path_action_hint(handle, pos, direction, transitions, distance_map)
            except Exception as e:
                print(f"[Warn] get: Fehler bei _shortest_path_action_hint: {e}")
                raw_features[1:4] = 0.0
            raw_features[4] = 1.0 if (decision_type & 4) else 0.0
            try:
                raw_features[5] = self._encode_detect_deadlock(self._detect_deadlock(handle, pos, direction))
            except Exception as e:
                print(f"[Warn] get: Fehler bei _detect_deadlock: {e}")
                raw_features[5] = 0.0
            raw_features[30] = 1.0 if ((decision_type & 2) or (decision_type & 4)) else 0.0
            all_distance = []
            for idx, a in enumerate(self.env.agents):
                apos = a.position if a.position is not None else a.initial_position
                adir = a.direction if a.direction is not None else a.initial_direction
                if apos is None or adir is None:
                    adist = np.inf
                else:
                    try:
                        adist = float(distance_map[a.handle, apos[0], apos[1], adir])
                    except Exception as e:
                        print(f"[Warn] get: Fehler bei distance_map für Agent {a.handle}: {e}")
                        adist = np.inf
                all_distance.append((a.handle, adist, idx))
            all_distance.sort(key=lambda x: (x[1], x[2]))
            value_to_rank = {}
            next_rank = 1
            handle_to_rank = {}
            for h, dist, _ in all_distance:
                if dist not in value_to_rank:
                    value_to_rank[dist] = next_rank
                    next_rank += 1
                handle_to_rank[h] = value_to_rank[dist]
            priority_rank = float(handle_to_rank.get(handle, next_rank)) / next_rank
            opp_agents = set()
            opp_agents.update(local_search_seen_agents)
            visited_type_2: set = set()
            visited_type_3_fwd: set = set()
            visited_type_3_bwd: set = set()
            for other in self.env.agents:
                if other.handle == handle:
                    continue
                other_pos = other.position if other.position is not None else other.initial_position
                if other_pos == pos:
                    opp_agents.add(other.handle)
            if decision_type & 2:
                for rel_dir in (-1, 0, 1):
                    abs_dir = (direction + rel_dir) % 4
                    base = 6 + (rel_dir + 1) * 8
                    if transitions[abs_dir]:
                        npos = get_new_position(pos, abs_dir)
                        try:
                            branch_dm = distance_map[handle, npos[0], npos[1], abs_dir]
                            branch_target_reachable = 1.0 if np.isfinite(branch_dm) else 0.0
                        except Exception as e:
                            print(f"[Warn] get: Fehler bei branch_dm: {e}")
                            branch_target_reachable = 0.0
                        try:
                            branch_dist, deadlock, switches, seen, abort, target_found, visited_type_2 = \
                                self._navigate_direction(handle, npos, abs_dir, target, False)
                        except Exception as e:
                            print(f"[Warn] get: Fehler bei _navigate_direction: {e}")
                            branch_dist = 0.0; deadlock = 0.0; switches = 0; seen = []; abort = 1.0; target_found = 0.0; visited_type_2 = set()
                        try:
                            branch_deadlock_ahead = self._detect_deadlock(handle, npos, abs_dir)
                        except Exception as e:
                            print(f"[Warn] get: Fehler bei branch_deadlock_ahead: {e}")
                            branch_deadlock_ahead = 0.0
                        opp_agents.update(seen)
                        branch_dist_norm = self._normalise_distance(branch_dist, max_dist)
                        switches_norm = self._normalise_count(switches)
                        progress_gain = float(np.clip(curr_dist_norm - branch_dist_norm, 0.0, 1.0))
                        branch_usable = 1.0 if (branch_target_reachable > 0.0 and branch_deadlock_ahead <= 0 and progress_gain > 0.0) else 0.0
                        raw_features[base + 0] = progress_gain
                        branch_deadlock_signal = max(float(deadlock), 2.0 if branch_deadlock_ahead > 0 else 0.0)
                        raw_features[base + 1] = self._encode_deadlock_signal(branch_deadlock_signal)
                        raw_features[base + 2] = switches_norm
                        raw_features[base + 3] = branch_dist_norm
                        raw_features[base + 4] = branch_target_reachable
                        raw_features[base + 5] = abort
                        raw_features[base + 6] = self._encode_detect_deadlock(branch_deadlock_ahead)
                        raw_features[base + 7] = branch_usable
                    else:
                        raw_features[base + 0] = 0.0
                        raw_features[base + 1] = 0.0
                        raw_features[base + 2] = 0.0
                        raw_features[base + 3] = 0.0
                        raw_features[base + 4] = 0.0
                        raw_features[base + 5] = 1.0
                        raw_features[base + 6] = 0.0
                        raw_features[base + 7] = 0.0
            forward_dir = fast_argmax(transitions)
            npos_fwd = get_new_position(pos, forward_dir)
            if decision_type & 4:
                merge_wait_urgency = 0.0
                try:
                    merge_wait_urgency = self._estimate_ttc_conflict_risk(handle, pos, direction, horizon=6)
                except Exception as e:
                    print(f"[Warn] get: Fehler bei _estimate_ttc_conflict_risk: {e}")
                try:
                    _, deadlock_fwd, switches_fwd, seen_fwd, abort_fwd, target_found_fwd, visited_type_3_fwd = \
                        self._navigate_direction(handle, npos_fwd, forward_dir, target, False)
                except Exception as e:
                    print(f"[Warn] get: Fehler bei _navigate_direction (fwd): {e}")
                    deadlock_fwd = 0.0; switches_fwd = 0; seen_fwd = []; abort_fwd = 1.0; target_found_fwd = 0.0; visited_type_3_fwd = set()
                try:
                    merge_deadlock_ahead_fwd = self._detect_deadlock(handle, npos_fwd, forward_dir)
                except Exception as e:
                    print(f"[Warn] get: Fehler bei merge_deadlock_ahead_fwd: {e}")
                    merge_deadlock_ahead_fwd = 0.0
                opp_agents.update(seen_fwd)
                raw_features[31] = self._encode_deadlock_signal(max(float(deadlock_fwd), 2.0 if merge_deadlock_ahead_fwd > 0 else 0.0))
                raw_features[32] = self._normalise_count(switches_fwd)
                raw_features[33] = float(np.clip(target_found_fwd, 0.0, 1.0))
                raw_features[34] = abort_fwd
                raw_features[35] = self._encode_detect_deadlock(merge_deadlock_ahead_fwd)
                raw_features[38] = float(np.clip(merge_wait_urgency, 0.0, 1.0))
                bwd_pos = None
                bwd_dir = None
                for d in range(4):
                    nd = (forward_dir + d) % 4
                    try:
                        nt = self.env.rail.get_transitions(*npos_fwd, nd)
                    except Exception as e:
                        print(f"[Warn] get: Fehler bei get_transitions (bwd): {e}")
                        continue
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
                    try:
                        _, deadlock_bwd, switches_bwd, seen_bwd, abort_bwd, target_found_bwd, visited_type_3_bwd = \
                            self._navigate_direction(handle, bwd_pos, bwd_dir, target, True)
                    except Exception as e:
                        print(f"[Warn] get: Fehler bei _navigate_direction (bwd): {e}")
                        deadlock_bwd = 0.0; switches_bwd = 0; seen_bwd = []; abort_bwd = 1.0; target_found_bwd = 0.0; visited_type_3_bwd = set()
                    try:
                        merge_deadlock_ahead_bwd = self._detect_deadlock(handle, bwd_pos, bwd_dir)
                    except Exception as e:
                        print(f"[Warn] get: Fehler bei merge_deadlock_ahead_bwd: {e}")
                        merge_deadlock_ahead_bwd = 0.0
                    opp_agents.update(seen_bwd)
                    raw_features[36] = self._encode_deadlock_signal(max(float(deadlock_bwd), 2.0 if merge_deadlock_ahead_bwd > 0 else 0.0))
                    raw_features[37] = self._normalise_count(switches_bwd)
                    raw_features[38] = float(np.clip(max(raw_features[38], float(target_found_bwd)), 0.0, 1.0))
                    raw_features[39] = abort_bwd
                    raw_features[40] = self._encode_detect_deadlock(merge_deadlock_ahead_bwd)
            state_value = int(agent.state.value)
            if 0 <= state_value <= 6:
                raw_features[41 + state_value] = 1.0
            # Last-action one-hot with behavior-aware fallback.
            # Priority:
            # 1) true saved_action from Flatland ActionSaver
            # 2) infer from movement + heading delta (old_direction -> direction)
            #    to avoid dead L/R/S features when ActionSaver is empty.
            sa = None
            on_map = getattr(agent, "position", None) is not None
            if on_map and agent.action_saver.is_action_saved:
                sa = int(agent.action_saver.saved_action)
            elif on_map:
                is_moving = bool(getattr(agent, "moving", False))
                if not is_moving:
                    sa = 4  # STOP_MOVING
                else:
                    old_dir = getattr(agent, "old_direction", None)
                    cur_dir = getattr(agent, "direction", None)
                    if old_dir is None or cur_dir is None:
                        sa = 2  # MOVE_FORWARD (safe default while moving)
                    else:
                        delta = (int(cur_dir) - int(old_dir)) % 4
                        if delta == 0:
                            sa = 2  # FORWARD
                        elif delta == 1:
                            sa = 3  # RIGHT
                        elif delta == 3:
                            sa = 1  # LEFT
                        else:
                            # U-turns are not action primitives; map to forward signal.
                            sa = 2
            # Off-map states stay neutral (all action slots 0) to avoid synthetic priors.
            if sa is not None and 0 <= sa <= 4:
                raw_features[48 + sa] = 1.0
            raw_features[53] = priority_rank
            curr_idx = self._cell_type_index_from_decision_type(decision_type)
            raw_features[54 + curr_idx] = 1.0
            _TR_SLOTS = {(1, 1): 59, (1, 2): 60, (1, 3): 61, (3, 1): 62, (2, 1): 63}
            next_decision_type = decision_type
            try:
                if decision_type & 8:
                    next_decision_type = 8
                elif decision_type == 1:
                    next_decision_type = self._decision_type_at_position(pos, direction, target)
                else:
                    if fast_count_nonzero(transitions) > 0:
                        forward_dir = fast_argmax(transitions)
                        next_pos = get_new_position(pos, forward_dir)
                        next_decision_type = self._decision_type_at_position(next_pos, forward_dir, target)
            except Exception as e:
                print(f"[Warn] get: Fehler bei next_decision_type: {e}")
                next_decision_type = decision_type
            next_idx = self._cell_type_index_from_decision_type(next_decision_type)
            tr_slot = _TR_SLOTS.get((curr_idx, next_idx), None)
            if tr_slot is not None:
                raw_features[tr_slot] = 1.0
            try:
                raw_features[65] = 1.0 if DecisionPointUtils.is_local_deadlock(self.env, agent, self.agent_map) else 0.0
            except Exception as e:
                print(f"[Warn] get: Fehler bei is_local_deadlock: {e}")
                raw_features[65] = 0.0
            # Soft marker for unreachable-from-current-state: keep information,
            # but avoid hard overrides that can destabilize PPO value/policy heads.
            if not curr_reachable:
                raw_features[5] = max(raw_features[5], 0.5)   # moderate local risk hint
                raw_features[53] = min(raw_features[53], 0.25)  # lower priority, not absolute zero
            visited: List = []
            for a in visited_type_2:
                visited.append(a[0])
            for a in visited_type_3_fwd:
                visited.append(a[0])
            for a in visited_type_3_bwd:
                visited.append(a[0])
            try:
                self.env.dev_obs_dict.update({handle: visited})
            except Exception as e:
                print(f"[Warn] get: Fehler bei dev_obs_dict.update: {e}")
            # Raw tree statistics für learnable tree_proj
            if tree_data:
                try:
                    _dl = [n["deadlock_risk"] for n in tree_data]
                    _cf = [min(1.0, len(n.get("agents_encountered", [])) / 2.0) for n in tree_data]
                    _br = [min(1.0, n.get("num_transitions", 1) / 3.0) for n in tree_data]
                    raw_features[66] = float(np.mean(_dl))
                    raw_features[67] = float(np.max(_dl))
                    raw_features[68] = float(np.mean(_cf))
                    raw_features[69] = float(np.mean(_br))
                except Exception as e:
                    print(f"[Warn] get: Fehler bei tree_data-Statistiken: {e}")
            agent.cur_opp_agent_handles = sorted(opp_agents)
            return (raw_features, agent.cur_opp_agent_handles)
        except Exception as e:
            print(f"[Warn] get: Schwerwiegender Fehler für Agent {handle}: {e}")
            return (raw_features, [])

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
            unreachable_here = bool(cur_dist == np.inf)

            if controller['count'] >= budget:
                return cur_dist, 0, num_switches, 1, -1
            if depth >= max_depth:
                return cur_dist, -1, num_switches, 1, -1
            if pos == target and not backward:
                return cur_dist, 0, num_switches, 0, 1
            if (pos, dirn) in controller['visited']:
                return cur_dist, -1, num_switches, 1, -1
            # Important: unreachable target state (np.inf) must NOT stop traversal,
            # because deadlock/conflict evidence can still exist on this branch.
            if unreachable_here:
                cur_dist = _UNREACHABLE

            controller['visited'].add((pos, dirn))
            controller['count'] += 1

            transitions = env.rail.get_transitions(*pos, dirn)

            if self.agent_map is not None:
                agent_idx = self.agent_map[pos]
                if agent_idx != -1 and agent_idx != handle:
                    controller['seen_agents'].add(agent_idx)
                    other_dir = env.agents[agent_idx].direction
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

    def _process_tree_data_with_lstm(self, tree_data: list) -> np.ndarray:
        """Verarbeitet Baumdaten via deterministischem TreeLSTM-Aggregator.

        Gibt einen Vektor der Länge ``hidden_dim`` zurück, dessen erster Wert
        (Index 0) das tiefengewichtete Deadlock-Risiko über den lokalen Suchbaum
        zusammenfasst.  Dieser Wert wird in Feature-Slot [64] geschrieben.
        """
        return self.tree_lstm.aggregate(tree_data)

    def _detect_deadlock(self, handle, pos, direction):
        """Detect confirmed corridor blockage before the next switch.

        This is broader than pure head-on detection: a same-direction queue is
        also treated as a deadlock if the blocking chain itself cannot clear
        before the next decision point.
        """
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
                    if DecisionPointUtils.is_opposite_direction(cur_dir, other.direction) or self._is_forward_only(cur_pos, cur_dir, cur_pos, other.direction):
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
                    if DecisionPointUtils.is_opposite_direction(ndir, other.direction):
                        conflict_hits += 1
                frontier.append((npos, ndir, depth + 1))

        return float(np.clip(conflict_hits / 3.0, 0.0, 1.0))

    def _is_forward_only(self, pos1, dir1, pos2, dir2) -> bool:
        t1 = self.env.rail.get_transitions(*pos1, dir1)
        t2 = self.env.rail.get_transitions(*pos2, dir2)
        return fast_count_nonzero(t1) == 1 and fast_count_nonzero(t2) == 1
