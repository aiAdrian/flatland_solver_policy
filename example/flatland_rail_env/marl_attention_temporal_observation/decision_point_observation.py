from flatland.core.env_observation_builder import ObservationBuilder
import numpy as np
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.step_utils.states import TrainState

class DecisionPointObservation(ObservationBuilder):
    """
    Observation builder focused on Flatland's three key decision points:
    1. Agent start (READY_TO_DEPART): Should the agent enter the board?
    2. At a switch: Path/routing decision (agent can branch).
    3. At a merge/crossing: One cell before a switch, cannot branch (merge/crossing logic).

    Encodes:
    - Decision type (start, switch, merge/crossing, or always-move)
    - Local cell features (is_switch, is_merge, can_branch, can_merge, can_cross)
    - Agent state, direction, position, and target
    - Optionally, temporal context (for use with temporal stacking)
    """

    def _navigate_direction(self, handle, start_pos, start_dir, target, max_steps=100):
        '''
        Navigiert ab start_pos/start_dir bis zum Ziel oder Deadlock.
        Gibt (dist, deadlock_flag, num_switches) zurück.
        Deadlock: 1, falls kein Weg zum Ziel gefunden werden kann.
        num_switches: Wie oft wurde der Pfad gewechselt (an Weichen).
        '''
        env = self.env
        distance_map = env.distance_map.get()
        agent_map = env.agent_map if hasattr(env, 'agent_map') else None
        pos = start_pos
        direction = start_dir
        steps = 0
        num_switches = 0
        visited = set()
        deadlock_flag = 0
        while steps < max_steps:
            if pos == target:
                return steps, 0, num_switches
            visited.add((pos, direction))
            transitions = env.rail.get_transitions(*pos, direction)
            # Prüfe entgegenkommende Agenten
            if agent_map is not None:
                agent_idx = agent_map[pos]
                if agent_idx != -1 and env.agents[agent_idx].direction == (direction + 2) % 4:
                    # Entgegenkommender Agent -> Backtrack
                    # Suche letzte Weiche rückwärts
                    back_pos, back_dir = pos, (direction + 2) % 4
                    found_switch = False
                    for _ in range(10):
                        back_trans = env.rail.get_transitions(*back_pos, back_dir)
                        if np.sum(back_trans) > 1:
                            found_switch = True
                            break
                        if back_trans[back_dir]:
                            back_pos = get_new_position(back_pos, back_dir)
                        else:
                            break
                    if found_switch:
                        num_switches += 1
                        # Versuche andere Richtung an der Weiche
                        for d in range(4):
                            if d != back_dir and back_trans[d]:
                                npos = get_new_position(back_pos, d)
                                if (npos, d) not in visited:
                                    # Rekursiver Versuch ab neuer Richtung
                                    sub_dist, sub_deadlock, sub_switches = self._navigate_direction(handle, npos, d, target, max_steps-steps)
                                    if sub_deadlock == 0:
                                        return steps + sub_dist, 0, num_switches + sub_switches
                        # Kein Weg gefunden
                        return -1, 1, num_switches
                    else:
                        return -1, 1, num_switches
            # Normale Fortsetzung auf dem kürzesten Pfad
                # Wähle Richtung mit minimaler Distanz zum Ziel
                min_dist = float('inf')
                best_dir = None
                for d in range(4):
                    if transitions[d]:
                        npos = get_new_position(pos, d)
                        dist = distance_map[handle, npos[0], npos[1], d]
                        if dist < min_dist:
                            min_dist = dist
                            best_dir = d
                if best_dir is None or min_dist == np.inf:
                    return -1, 1, num_switches
                pos = get_new_position(pos, best_dir)
                direction = best_dir
                steps += 1
            return -1, 1, num_switches
    """
    Observation builder focused on Flatland's three key decision points:
    1. Agent start (READY_TO_DEPART): Should the agent enter the board?
    2. At a switch: Path/routing decision (agent can branch).
    3. At a merge/crossing: One cell before a switch, cannot branch (merge/crossing logic).

    Encodes:
    - Decision type (start, switch, merge/crossing, or always-move)
    - Local cell features (is_switch, is_merge, can_branch, can_merge, can_cross)
    - Agent state, direction, position, and target
    - Optionally, temporal context (for use with temporal stacking)
    """
    def __init__(self):
        super().__init__()
        self.env = None
        self.switchAnalyser = None
        self.feature_len = 30
        print(">> DecisionPointObservation loaded.")

    def set_env(self, env):
        self.env = env
        self.switchAnalyser = None

    def reset(self):
        self.switchAnalyser = None

    @staticmethod
    def getObservationSize() -> int:
        return 30

    def get(self, handle: int = 0):
        agent = self.env.agents[handle]  
        pos = agent.position if agent.position is not None else agent.initial_position
        dir = agent.direction if agent.direction is not None else agent.initial_direction
        target = agent.target
        if pos is None or target is None:
            return (np.zeros(self.feature_len, dtype=np.float32)-1, [])

        if self.switchAnalyser is None:
            from flatland_railway_extension.RailroadSwitchAnalyser import RailroadSwitchAnalyser
            self.switchAnalyser = RailroadSwitchAnalyser(self.env)

        agent_at_switch, agent_near_switch, switch_cell, near_switch_cell = \
            self.switchAnalyser.check_agent_decision(position=pos, direction=dir)

        distance_map = self.env.distance_map.get()

        agent_idx = self.env.agent_map[pos] if hasattr(self.env, 'agent_map') and self.env.agent_map is not None else -1
        if agent_idx != -1:
            pass

        if agent.state.name == "READY_TO_DEPART":
            decision_type = 1.0
        elif agent_at_switch:
            # agent is currently at a switch (branching logic)
            decision_type = 2.0
        elif near_switch_cell and not agent_near_switch:
            # agent is one cell before a switch (merge/crossing logic)
            decision_type = 3.0
        else:
            decision_type = 0.0
        
        features = np.zeros(self.feature_len, dtype=np.float32)

        features[0] = decision_type

        # Basic agent info -> shortes path hint -> which agent goes to target as fast as possible (dummy logic for now, can be improved)
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
        features[1:4] = best_hint

        # decision_type 2: auf Weiche -> für jede Richtung [dist, deadlock, switches]
        if decision_type == 2.0:
            rel_dirs = [(-1) % 4, 0, 1, 2]  # left, forward, right, reverse
            for i, rel_dir in enumerate(rel_dirs):
                abs_dir = (dir + rel_dir) % 4
                if transitions[abs_dir]:
                    npos = get_new_position(pos, abs_dir)
                    dist, deadlock, switches = self._navigate_direction(handle, npos, abs_dir, target)
                else:
                    dist, deadlock, switches = -1, 1, 0
                features[4 + i*3] = dist
                features[4 + i*3 + 1] = deadlock
                features[4 + i*3 + 2] = switches
        if decision_type == 3.0:
            # Vor Weiche/Merge: Forward-Analyse (Einmündung/Kreuzung)
            forward_dir = dir
            reverse_dir = (dir + 2) % 4
            forward_free = 1.0
            forward_agent = 0.0
            forward_deadlock = 0.0
            wait_flag = 0.0
            # Prüfe Feld vorwärts
            npos = get_new_position(pos, forward_dir)
            if hasattr(self.env, 'agent_map') and self.env.agent_map is not None:
                agent_on_next = self.env.agent_map[npos] if npos is not None else -1
                if agent_on_next != -1:
                    forward_free = 0.0
                    other_agent = self.env.agents[agent_on_next]
                    # Kommt Agent entgegen?
                    if other_agent.direction == reverse_dir:
                        forward_agent = 1.0
                        forward_deadlock = 1.0
            # Prüfe, ob aus Gegenrichtung ein Agent kommt, der Vorrang haben sollte
            npos_rev = get_new_position(pos, reverse_dir)
            agent_on_prev = self.env.agent_map[npos_rev] if npos_rev is not None and hasattr(self.env, 'agent_map') and self.env.agent_map is not None else -1
            if agent_on_prev != -1:
                other_agent = self.env.agents[agent_on_prev]
                # Kommt Agent auf mich zu?
                if other_agent.direction == forward_dir:
                    wait_flag = 1.0
            # Features setzen
            features[16] = forward_free
            features[17] = forward_agent
            features[18] = forward_deadlock
            features[19] = wait_flag
        return (features, [])

    def get_many(self, handles: list = None):
        if handles is None:
            handles = list(range(len(self.env.agents)))
        result = []
        for handle in handles:
            obs_self, obs_others = self.get(handle)
            result.append((obs_self, obs_others))
        return result

    @staticmethod
    def get_decision_point_observation(env, handle, switchAnalyser, walker, max_path_length, lookahead_cost_limit, max_agent_dist):
        agent = env.agents[handle]
        pos = agent.position if agent.position is not None else agent.initial_position
        direction = agent.direction if agent.direction is not None else agent.initial_direction
        if pos is None or direction is None or not agent.state.is_on_map_state():
            # Agent ist nicht auf der Map
            return np.zeros(30, dtype=np.float32), []

        # Beispielhafte Feature-Berechnung (hier kannst du deine Logik anpassen)
        from .experimental_observation import ExperimentalObservation
        # Korrigiere: Features-Länge auf 30 (wie erwartet)
        features = np.zeros(30, dtype=np.float32)
        features[0] = float(pos[0])
        features[1] = float(pos[1])
        features[2] = float(direction)
        features[3] = float(agent.state.value)
        features[4] = float(handle)
        # Dummy: Rest mit 0 (oder nach Bedarf weitere sinnvolle Features)
        # ... weitere Feature-Berechnung nach Bedarf ...

        # Dummy-opp_agents-Liste (hier ggf. echte Gegnerlogik einbauen)
        opp_agents = []
        return features, opp_agents
