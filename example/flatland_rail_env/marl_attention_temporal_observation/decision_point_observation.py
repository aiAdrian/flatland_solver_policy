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

    def _is_in_grid(self, pos):
        if pos is None:
            return False
        env = self.env
        rows, cols = env.height, env.width if hasattr(env, 'height') and hasattr(env, 'width') else env.rail.grid.shape
        r, c = pos
        return 0 <= r < rows and 0 <= c < cols

    def _navigate_direction(self, handle, start_pos, start_dir, target, max_steps=100):
        '''
        Navigiert ab start_pos/start_dir bis zum Ziel oder Deadlock.
        Gibt (dist, deadlock_flag, num_switches, seen_agents) zurück.
        Deadlock: 1, falls kein Weg zum Ziel gefunden werden kann.
        num_switches: Wie oft wurde der Pfad gewechselt (an Weichen).
        seen_agents: Liste aller fremden Agenten, die auf dem Pfad gesehen wurden.
        '''
        env = self.env
        distance_map = env.distance_map.get()
        agent_map = env.agent_map if hasattr(env, 'agent_map') else None
        pos = start_pos
        direction = start_dir
        steps = 0
        num_switches = 0
        visited = set()
        seen_agents = set()
        deadlock_flag = 0
        while steps < max_steps:
            if pos == target:
                return steps, 0, num_switches, list(seen_agents)
            if not self._is_in_grid(pos):
                return -1, 1, num_switches, list(seen_agents)
            visited.add((pos, direction))
            transitions = env.rail.get_transitions(*pos, direction)
            # Prüfe entgegenkommende Agenten
            if agent_map is not None:
                agent_idx = agent_map[pos]
                if agent_idx != -1 and agent_idx != handle:
                    seen_agents.add(agent_idx)
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
                                    sub_dist, sub_deadlock, sub_switches, sub_seen = self._navigate_direction(handle, npos, d, target, max_steps-steps)
                                    seen_agents.update(sub_seen)
                                    if sub_deadlock == 0:
                                        return steps + sub_dist, 0, num_switches + sub_switches, list(seen_agents)
                        return -1, 1, num_switches, list(seen_agents)
                    else:
                        return -1, 1, num_switches, list(seen_agents)
            # Normale Fortsetzung auf dem kürzesten Pfad
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
                return -1, 1, num_switches, list(seen_agents)
            pos = get_new_position(pos, best_dir)
            direction = best_dir
            steps += 1
        return -1, 1, num_switches, list(seen_agents)
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

        opp_agents = set()
        # decision_type 2: auf Weiche -> für jede Richtung [dist, deadlock, switches]
        if decision_type == 2.0:
            rel_dirs = [(-1) % 4, 0, 1, 2]  # left, forward, right, reverse
            for i, rel_dir in enumerate(rel_dirs):
                abs_dir = (dir + rel_dir) % 4
                if transitions[abs_dir]:
                    npos = get_new_position(pos, abs_dir)
                    dist, deadlock, switches, seen = self._navigate_direction(handle, npos, abs_dir, target)
                    opp_agents.update(seen)
                else:
                    dist, deadlock, switches = -1, 1, 0
                features[4 + i*3] = dist
                features[4 + i*3 + 1] = deadlock
                features[4 + i*3 + 2] = switches
        if decision_type == 3.0:
            # Vor Weiche/Merge: Forward- und Backward-Analyse mit _navigate_direction
            forward_dir = dir
            reverse_dir = (dir + 2) % 4
            # Forward: 1 Schritt vorwärts
            npos_fwd = get_new_position(pos, forward_dir)
            if npos_fwd is not None and self._is_in_grid(npos_fwd):
                dist_fwd, deadlock_fwd, switches_fwd, seen_fwd = self._navigate_direction(handle, npos_fwd, forward_dir, target)
                opp_agents.update(seen_fwd)
                forward_free = 1.0 if dist_fwd != -1 else 0.0
                forward_agent = 1.0 if len(seen_fwd) > 0 else 0.0
                forward_deadlock = float(deadlock_fwd)
            else:
                forward_free = 0.0
                forward_agent = 0.0
                forward_deadlock = 1.0
            # Backward: 1 Schritt rückwärts
            npos_bwd = get_new_position(pos, reverse_dir)
            wait_flag = 0.0
            if npos_bwd is not None and self._is_in_grid(npos_bwd):
                dist_bwd, deadlock_bwd, switches_bwd, seen_bwd = self._navigate_direction(handle, npos_bwd, reverse_dir, target)
                opp_agents.update(seen_bwd)
                if len(seen_bwd) > 0:
                    wait_flag = 1.0
            # Features setzen
            features[16] = forward_free
            features[17] = forward_agent
            features[18] = forward_deadlock
            features[19] = wait_flag
        return (features, list(opp_agents))

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
