from flatland.core.env_observation_builder import ObservationBuilder
import numpy as np
from flatland_railway_extension.RailroadSwitchAnalyser import RailroadSwitchAnalyser
from .walk_to_next_decision_point import WalkToNextDecisionPoint
from .experimental_observation import ExperimentalObservation
from flatland.core.grid.grid4_utils import get_new_position

class SimplifiedPathThreeTierObservation(ObservationBuilder):
    """
    Observation mit drei Pfad-Teilen (KORRIGIERT):
    - A: Links (nur aktiv bei Switch und wenn Pfad nach links existiert, sonst Nullvektor)
    - B: Forward (immer aktiv, enthält Features für geradeaus)
    - C: Rechts (nur aktiv bei Switch und wenn Pfad nach rechts existiert, sonst Nullvektor)
    Alle Teile haben identische Feature-Länge. Header enthält State-Infos.
    """
    def __init__(self):
        super().__init__()
        self.env = None
        self.switchAnalyser = None
        self.walker = None
        self.feature_len = 16  # Beispiel: 16 Features pro Pfad (anpassbar)
        self.header_len = 9    # z.B. State, Richtung, etc. + nbr_agents
        print(">> SimplifiedPathThreeTierObservation loaded.")

    def set_env(self, env):
        self.env = env
        self.switchAnalyser = None
        self.walker = None

    def reset(self):
        self.switchAnalyser = None
        self.walker = None

    @staticmethod
    def getObservationSize() -> int:
        # Header + 3 Pfade je feature_len
        return 9 + 3 * 16

    def get(self, handle: int = 0):
        agent = self.env.agents[handle]
        pos, dir = ExperimentalObservation.get_pos_dir(agent)
        target = agent.target
        if pos is None or target is None:
            return (np.zeros(self.getObservationSize(), dtype=np.float32)-1, [])

        # Initialisiere Analyser erst beim ersten Zugriff
        if self.switchAnalyser is None:
            self.switchAnalyser = RailroadSwitchAnalyser(self.env)
        if self.walker is None:
            self.walker = WalkToNextDecisionPoint(self.env)

        # Bestimme, welche Richtung (l, f, r) den kürzesten Pfad zum Ziel hat
        transitions = self.env.rail.get_transitions(*pos, dir)
        best_hint = [0.0, 0.0, 0.0]  # [l, f, r]
        if target is not None:
            min_dist = float('inf')
            best_idx = -1
            for idx, rel_dir in enumerate([(-1) % 4, 0, 1]):
                abs_dir = (dir + rel_dir) % 4
                if transitions[abs_dir]:
                    npos = get_new_position(pos, abs_dir)
                    distance_map = self.env.distance_map.get()
                    dist = distance_map[handle, npos[0], npos[1], abs_dir]
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = idx
            if best_idx != -1:
                best_hint[best_idx] = 1.0
        nbr_agents = float(len(self.env.agents)) if hasattr(self.env, 'agents') else -1.0
        header = [float(agent.state.value), float(dir), float(agent.handle), float(pos[0]), float(pos[1]), float(target[0]), float(target[1]), nbr_agents] + best_hint

        def path_features(start_pos, start_dir, max_steps=64, toleranz_max_dist_step_diff=5):
            f = np.zeros(self.feature_len, dtype=np.float32)
            if start_pos is None or target is None:
                return f
            transitions = self.env.rail.get_transitions(*start_pos, start_dir)
            distance_map = self.env.distance_map.get()
            h, w = self.env.height, self.env.width
            handle = agent.handle
            if 0 <= start_pos[0] < h and 0 <= start_pos[1] < w:
                dist = distance_map[handle, start_pos[0], start_pos[1], start_dir]
            else:
                dist = np.inf
            f[0] = dist
            f[1] = 0.0  # Entferne Switch-Flag, da es nicht benötigt wird

            pos = start_pos
            direction = start_dir
            # Erster Schritt: gehe auf das nächste Feld in die Richtung, falls möglich
            if transitions[direction]:
                pos = get_new_position(pos, direction)
            else:
                found = False
                for d in range(4):
                    if transitions[d]:
                        pos = get_new_position(pos, d)
                        direction = d
                        found = True
                        break
                if not found:
                    return f

            agent_count = 0
            opp_agent_count = 0
            deadlock_found = 0
            switch_found = 0
            deadlock_in_path = 0
            target_found = 0
            crowd = 0
            path_len = 0
            block_flag = 0.0
            near_switch = 0.0
            near_switch_cell = 0.0
            steps = 0
            greedy_mode = False

            # DEADLOCK-Feature: Gibt es einen entgegenkommenden Agenten bis zum nächsten Switch, der nicht ausweichen kann?
            deadlock_on_path = 0.0
            deadlock_checked = False
            while steps < max_steps:
                if pos == target:
                    target_found = 1
                    break
                transitions = self.env.rail.get_transitions(*pos, direction)
                if np.sum(transitions) == 0:
                    break
                # Agenten auf Feld?
                agent_idx = self.env.agent_map[pos] if hasattr(self.env, 'agent_map') and self.env.agent_map is not None else -1
                if agent_idx != -1 and agent_idx != handle:
                    agent_count += 1
                    other_dir = self.env.agents[agent_idx].direction
                    # Prüfe: Kommt Agent entgegen?
                    if (other_dir + 2) % 4 == direction:
                        opp_agent_count += 1
                        # Deadlock-Prüfung: Gibt es bis zum nächsten Switch für den anderen Agenten eine Ausweichmöglichkeit?
                        # Wir laufen ab hier bis zum nächsten Switch rückwärts entlang der Richtung des anderen Agenten
                        opp_pos = pos
                        opp_dir = other_dir
                        found_switch = False
                        for _ in range(32):
                            opp_trans = self.env.rail.get_transitions(*opp_pos, opp_dir)
                            if np.sum(opp_trans) > 2:
                                found_switch = True
                                break
                            # Weiterlaufen
                            if opp_trans[opp_dir]:
                                opp_pos = get_new_position(opp_pos, opp_dir)
                            else:
                                found = False
                                for d in range(4):
                                    if opp_trans[d]:
                                        opp_pos = get_new_position(opp_pos, d)
                                        opp_dir = d
                                        found = True
                                        break
                                if not found:
                                    break
                        if not found_switch:
                            deadlock_on_path = 1.0
                            deadlock_checked = True
                            break
                # Deadlock?
                legal_moves = [self.env.rail.get_transitions(*pos, (direction + a) % 4) for a in [-1, 0, 1]]
                if sum([np.count_nonzero(m) for m in legal_moves]) == 0:
                    deadlock_found = 1
                # Switch?
                at_switch, near_switch_flag, at_switch_cell, near_switch_cell_flag = self.switchAnalyser.check_agent_decision(position=pos, direction=direction)
                if at_switch or at_switch_cell:
                    switch_found = 1
                    # Ab jetzt greedy shortest path zum Ziel
                    greedy_mode = True
                    break  # Stoppe bei Switch, Deadlock-Check bis hier
                if near_switch_flag:
                    near_switch = 1.0
                if near_switch_cell_flag:
                    near_switch_cell = 1.0
                # Crowd
                y, x = pos
                h, w = self.env.height, self.env.width
                r = 2
                y_min, y_max = max(0, y - r), min(h, y + r + 1)
                x_min, x_max = max(0, x - r), min(w, x + r + 1)
                crowd += np.sum(self.env.agent_map[y_min:y_max, x_min:x_max] != -1) - 1 if hasattr(self.env, 'agent_map') and self.env.agent_map is not None else 0
                # Blockiert?
                if agent_idx != -1 and agent_idx != handle and self.env.agents[agent_idx].state == 2:  # TrainState.STOPPED
                    block_flag = 1.0

                # Weiterlaufen
                if greedy_mode:
                    min_dist = np.inf
                    best_dirs = []
                    dists = {}
                    for d in range(4):
                        if transitions[d]:
                            npos = get_new_position(pos, d)
                            if 0 <= npos[0] < h and 0 <= npos[1] < w:
                                d_dist = distance_map[handle, npos[0], npos[1], d]
                            else:
                                d_dist = np.inf
                            dists[d] = d_dist
                            if d_dist < min_dist:
                                min_dist = d_dist
                    for d, d_dist in dists.items():
                        if d_dist - min_dist <= toleranz_max_dist_step_diff:
                            best_dirs.append(d)
                    if direction in best_dirs:
                        chosen_dir = direction
                    else:
                        chosen_dir = best_dirs[0] if best_dirs else None
                    if chosen_dir is not None:
                        pos = get_new_position(pos, chosen_dir)
                        direction = chosen_dir
                    else:
                        break
                else:
                    if transitions[direction]:
                        pos = get_new_position(pos, direction)
                    else:
                        found = False
                        for d in range(4):
                            if transitions[d]:
                                pos = get_new_position(pos, d)
                                direction = d
                                found = True
                                break
                        if not found:
                            break
                path_len += 1
                steps += 1
            f[2] = agent_count
            f[3] = opp_agent_count
            f[4] = deadlock_found
            f[5] = switch_found
            f[6] = deadlock_on_path  # Deadlock-Feature: 1.0 falls Deadlock bis Switch
            f[7] = near_switch
            f[8] = near_switch_cell
            f[9] = crowd / max(1, path_len)
            f[10] = path_len
            f[11] = target_found
            f[12] = block_flag
            # f[13], f[14], f[15] = 0.0 (Padding)
            return f

        # KORRIGIERT: A=links, B=forward, C=rechts
        a = np.zeros(self.feature_len, dtype=np.float32)  # links
        b = np.zeros(self.feature_len, dtype=np.float32)  # forward
        c = np.zeros(self.feature_len, dtype=np.float32)  # rechts

        transitions = self.env.rail.get_transitions(*pos, dir)
        is_switch = np.sum(transitions) > 2
        # Links
        left_dir = (dir - 1) % 4
        if is_switch and transitions[left_dir]:
            a = path_features(pos, left_dir)
        # Forward immer gesetzt, wenn möglich
        if transitions[dir]:
            b = path_features(pos, dir)
        # Rechts
        right_dir = (dir + 1) % 4
        if is_switch and transitions[right_dir]:
            c = path_features(pos, right_dir)

        d = np.zeros(self.feature_len, dtype=np.float32)  # Path D
        # Path D nur aktivieren, wenn near_switch_cell_flag und nächstes Rückwärtsfeld frei
        _, _, _, near_switch_cell_flag = self.switchAnalyser.check_agent_decision(position=pos, direction=dir)
        reverse_dir = (dir + 2) % 4
        next_pos = get_new_position(pos, reverse_dir)
        agent_on_next = False
        if next_pos is not None and hasattr(self.env, 'agent_map') and self.env.agent_map is not None:
            agent_on_next = self.env.agent_map[next_pos] != -1
        if near_switch_cell_flag and not agent_on_next:
            d = self.path_features_d(pos, dir)

        obs = np.concatenate([header, a, b, c, d])
        return (obs, [])
    
    def path_features_d(self, start_pos, start_dir, max_steps=64):
        """
        Berechnet die Features für Path D (Rückfluss).
        - Zählt Agenten, die entlang des Rückflusses kommen.
        - Schätzt die verbleibende Pfaddistanz der Agenten bis zu ihrem Ziel (statt Luftlinie).
        Die Richtung wird korrekt um 180° gedreht (Rückweg).
        Fügt Grenzprüfung für das Grid ein.
        """
        f = np.zeros(self.feature_len, dtype=np.float32)
        if start_pos is None:
            return f

        h, w = self.env.height, self.env.width
        pos = start_pos
        direction = (start_dir + 2) % 4  # 180° gedreht
        steps = 0
        agent_count = 0
        total_path_dist_to_target = 0.0
        distance_map = self.env.distance_map.get()

        def in_bounds(p):
            return 0 <= p[0] < h and 0 <= p[1] < w

        while steps < max_steps:
            if not in_bounds(pos):
                break
            transitions = self.env.rail.get_transitions(*pos, direction)
            if np.sum(transitions) == 0:
                break

            # Agenten auf dem aktuellen Feld zählen
            agent_idx = self.env.agent_map[pos] if hasattr(self.env, 'agent_map') and self.env.agent_map is not None else -1
            if agent_idx != -1:
                agent_count += 1
                agent = self.env.agents[agent_idx]
                # Nutze Pfaddistanz statt Luftlinie
                agent_dir = agent.direction if agent.direction is not None else agent.initial_direction
                path_dist = distance_map[agent_idx, pos[0], pos[1], agent_dir]
                if path_dist != np.inf:
                    total_path_dist_to_target += path_dist

            # Weiter entlang des Rückflusses gehen: immer in die Richtung (direction) weiterlaufen
            next_pos = get_new_position(pos, direction)
            if next_pos == pos or next_pos is None or not in_bounds(next_pos):
                break
            pos = next_pos
            # Richtung bleibt gleich (immer rückwärts)
            steps += 1

        f[0] = agent_count
        f[1] = total_path_dist_to_target / max(1, agent_count)  # Durchschnittliche Pfaddistanz zum Ziel
        # Padding für restliche Features
        return f

    def get_many(self, handles: list = None):
        if handles is None:
            handles = list(range(len(self.env.agents)))
        result = []
        for handle in handles:
            obs_self, obs_others = self.get(handle)
            # obs_others ist immer leer, aber für Kompatibilität mit TemporalMultiAgentObservation
            result.append((obs_self, obs_others))
        return result