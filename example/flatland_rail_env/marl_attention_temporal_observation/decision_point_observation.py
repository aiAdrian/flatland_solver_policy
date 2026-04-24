from flatland.envs.fast_methods import fast_count_nonzero, fast_argmax
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
    
    def __init__(self):
        super().__init__()
        self.env = None
        self.feature_len = DecisionPointObservation.getObservationSize()
        print(">> DecisionPointObservation loaded.")

    def set_env(self, env):
        self.env = env 

    def reset(self):
         self.agent_map = np.zeros((self.env.height, self.env.width), dtype=np.int32) - 1

    @staticmethod
    def getObservationSize() -> int:
        # Ursprüngliche Feature-Länge + 6 neue Features
        return 42

    def get(self, handle: int = 0): 
        # init features
        features = np.zeros(self.feature_len, dtype=np.float32)

        all_visited = set()

        # Get base information about the agent and its environment
        agent = self.env.agents[handle]  

        pos = agent.position if agent.position is not None else agent.initial_position
        dir = agent.direction if agent.direction is not None else agent.initial_direction
        target = agent.target
        if pos is None or target is None:
            return (features - 1, [])

        # retrieve distance map for pathfinding features
        distance_map = self.env.distance_map.get()
        curr_dist = self.env.distance_map.get()[handle, pos[0], pos[1], dir]  
        if curr_dist == np.inf:
            return (features - 1, [])

        # get current possible transistions
        transitions = self.env.rail.get_transitions(*pos, dir)
        merge_switch = False
        for nd in range(4):  # Nur bei genau einem möglichen Übergang prüfen
            if transitions[nd]:
                next_pos = get_new_position(pos, nd)  
                for d in range(4):  
                    next_transitions = self.env.rail.get_transitions(*next_pos, d)        
                    if fast_count_nonzero(next_transitions) > 1:  # Wenn der nächste Übergang ein Switch ist
                        if d != nd:
                            merge_switch = True

        # agent is traveling on a normal track segment (no decision point)
        decision_type = 0

        # classify decision point type
        if agent.state.name == "READY_TO_DEPART":
            decision_type = 1
        else:
            if fast_count_nonzero(transitions) > 1:
                # agent is currently at a switch (branching logic)
                # 
                #                            / *** [ ] *** [ ] *** 
                # --- [ ] ---  [switch: agent] *** [ ] *** [ ] *** 
                # 
                # > agent travel direction 
                # * path to search (shortest path first) if blocked otherpath ,... -> local serach (~one path in the tree)
                decision_type += 2

            if merge_switch:
                # agent is one cell before a (merge/crossing) switch 
                # 
                # *** [ ] *** [     ] *** \  
                # --- [ ] --- [agent] --- [switch] *** [ ] *** 
                # 
                # > agent travel direction 
                # * path to search 
                decision_type += 4

        if agent.state.name == "DONE":
            # agent has completed its journey
            decision_type = 8

               
        

        # Feature 0: decision_type
        features[0] = decision_type
        # Feature 1,2,3 - shortest path action hint (left, forward, right) along the optimal path to target (only for switch and merge/crossing decision points)
        features[1:4] = self._shortest_path_action_hint(handle, pos, dir, transitions, distance_map)

        opp_agents = set()
        abort_flag = 0
        visited_type_2 = set()
        visited_type_3_fwd = set()
        visited_type_3_bwd = set()

        # decision_type 2: agent on a switch -> for each direction [dist, deadlock, switches]
        if decision_type & 2:
            rel_dirs = [-1, 0, 1]  # left, forward, right, reverse
            for i, rel_dir in enumerate(rel_dirs):
                abs_dir = (dir + rel_dir) % 4
                base = 4 + (rel_dir + 1) * 6
                if transitions[abs_dir]:
                    npos = get_new_position(pos, abs_dir)
                    dist, deadlock, switches, seen, abort, target_found, visited_type_2 = \
                        self._navigate_direction(handle, npos, abs_dir, target, False)
                    opp_agents.update(seen)
                    abort_flag = max(abort_flag, abort) 
                else:
                    dist, deadlock, switches, abort, target_found  = -1, -1, -1, -1, -1

                features[base] = curr_dist      # 4,10,16: dist
                features[base + 1] = deadlock   # 5,11,17: deadlock
                features[base + 2] = switches   # 6,12,18: switches
                features[base + 3] = dist # 7,13,19: dist
                features[base + 4] = target_found  # 8,14,20: target_found_flag
                features[base + 5] = abort  # 9,15,21: abort_flag

        # decision_type 3: agent near a merging switch 
        # -  look forward and backward for crossing and ordering (priority) logic 
        #    (e.g., if forward path is blocked by another agent, can we merge/cross? 
        #    If backward path is blocked, do we have priority?)
        # - cannot branch, but can merge/cross) -> for each direction 
        #   [dist, deadlock, switches, delta_dist, target_found, abort]
        if decision_type & 4: 
            # ----------------------------------------------
            # Forward: 1 step forward (merge/crossing logic)
            # ----------------------------------------------
            # 
            # --- [ ] --- [       ] --- \  
            # --- [ ] --- [agent >] --- [switch] *** [< other agent] ***
            # 
            # > agent travel direction   
            # < other agent's travel direction which agent must let pass (deadlock avoidance -> no options)
            # * path to _navigate forward is blocked by other agent -> can we merge/cross? (switch logic, deadlock avoidance -> maybe options)

            # 1. Alle möglichen Ausgänge für die aktuelle Blickrichtung holen
            transitions = self.env.rail.get_transitions(*pos, dir)
            # 2. Die einzige mögliche Richtung finden (ndir)
            # Da es nur eine Option gibt, ist nur ein Index in transitions == 1
            forward_dir = np.argmax(transitions) 

            
            # 3. Die neue Position basierend auf dieser Richtung berechnen
            npos_fwd = get_new_position(pos, forward_dir)

            dist, deadlock_fwd, switches_fwd, seen_fwd, abort_fwd, target_found_fwd, visited_type_3_fwd = \
                self._navigate_direction(handle, npos_fwd, forward_dir, target, False)
            opp_agents.update(seen_fwd)
            abort_flag = max(abort_flag, abort_fwd) 
 
            features[22] = deadlock_fwd      # 22: deadlock_fwd
            features[23] = switches_fwd      # 23: switches_fwd
            features[24] = target_found_fwd  # 24: target_found_fwd
            features[25] = abort_fwd         # 25: abort_flag_fwd
            
            # ----------------------------------------------
            # Backward: 1 step backward (merge/crossing logic) 
            # ----------------------------------------------
            # 
            # *** [ ] *** [other agent >] *** \  
            # --- [ ] --- [agent >      ] --- [switch] --- [ ] --- 
            # 
            # > agent travel direction 
            # > other agent travel direction 
            # decision to take agent before other agent or other agent before agent
            # * path to _navigate backward is blocked by other agent -> do we have priority? (switch logic, deadlock avoidance -> maybe options)
            bwd_pos = None
            bwd_dir = None

            for d in range(1,4):
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
                _, deadlock_bwd, switches_bwd, seen_bwd, abort_bwd, target_found_bwd, visited_type_3_bwd = \
                    self._navigate_direction(handle, bwd_pos, bwd_dir, target, True)
                opp_agents.update(seen_bwd) 

                features[26] = deadlock_bwd      # 26: deadlock_bwd
                features[27] = switches_bwd      # 27: switches_bwd
                features[28] = target_found_bwd  # 28: target_found_bwd
                features[29] = abort_bwd         # 29: abort_flag_bwd

        # agent.state => 
        #   TrainState.WAITING
        #     READY_TO_DEPART = 1
        #     MALFUNCTION_OFF_MAP = 2
        #     MOVING = 3
        #     STOPPED = 4
        #     MALFUNCTION = 5
        #     DONE = 6
        features[30 + agent.state.value] = 1.0 # 30-36: one-hot encoding des Agentenzustands (READY_TO_DEPART, MALFUNCTION_OFF_MAP, MOVING, STOPPED, MALFUNCTION, DONE)

        #     DO_NOTHING = 0  # implies change of direction in a dead-end!
        #     MOVE_LEFT = 1
        #     MOVE_FORWARD = 2
        #     MOVE_RIGHT = 3
        #     STOP_MOVING = 4
        if agent.action_saver.is_action_saved:
            features[37 + agent.action_saver.saved_action] = 1.0 # 37-41: one-hot encoding der zuletzt gespeicherten Aktion (MOVE_LEFT, MOVE_FORWARD, MOVE_RIGHT, STOP_MOVING)

        if True: 
            visited = []
            for a in visited_type_2:
                visited.append(a[0])
            for a in visited_type_3_fwd:
                visited.append(a[0])
            for a in visited_type_3_bwd:
                visited.append(a[0])
            self.env.dev_obs_dict.update({handle: visited})
        else:
            if handle == 0: 
                visited = []
                for a in visited_type_2:
                    visited.append(a[0])
                self.env.dev_obs_dict.update({0: visited}) 
                visited = []
                for a in visited_type_3_fwd:
                    visited.append(a[0])
                self.env.dev_obs_dict.update({1: visited}) 
                visited = []
                for a in visited_type_3_bwd: 
                    visited.append(a[0])
                self.env.dev_obs_dict.update({2: visited})

        agent.cur_opp_agent_handles = list(opp_agents)  # Speichere die Gegner-Handles im Agentenobjekt
        return (features, agent.cur_opp_agent_handles)

    def get_many(self, handles: list = None):
        if handles is None:
            handles = list(range(len(self.env.agents)))
        result = []

        self.agent_map = np.zeros((self.env.height, self.env.width), dtype=np.int32) - 1
        for agent in self.env.agents:
            if agent.position is not None: 
                self.agent_map[agent.position] = agent.handle   

        for agent in self.env.agents: 
            if not hasattr(agent, 'opp_agent_handles'):
                agent.opp_agent_handles = []  # Initialisiere leere Liste für Gegner-Handles
            if not hasattr(agent, 'cur_opp_agent_handles'):
                agent.cur_opp_agent_handles = []  # Initialisiere leere Liste für Gegner-Handles

        for handle in handles:
            obs_self, obs_others = self.get(handle)
            result.append((obs_self, obs_others))

        for agent in self.env.agents:
            agent.opp_agent_handles = agent.cur_opp_agent_handles  # Aktualisiere die Gegner-Handles im Agentenobjekt

        return result


    def _shortest_path_action_hint(self, handle, pos, dir, transitions, distance_map):
        """
        Berechnet die beste Aktionsrichtung (left, forward, right) entlang des kürzesten Pfads zum Ziel.
        Gibt einen one-hot Vektor zurück, der die beste Richtung markiert.
        """
        best_hint = [0.0, 0.0, 0.0]  # [l, f, r]
        min_dist = np.inf
        best_idx = 1  # Default: forward
        idx = 0
        for ndir in [(dir + i) % 4 for i in range(-1, 2)]:
            if transitions[ndir]:
                npos = get_new_position(pos, ndir)
                dist = distance_map[handle, npos[0], npos[1], ndir]
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
            idx += 1
        if best_idx != -1:
            best_hint[best_idx] = 1.0

        return best_hint

    def _navigate_direction(self, handle, start_pos, start_dir, target, backward_trace, max_steps=100):
        '''
        Navigiert ab start_pos/start_dir bis zum Ziel oder Deadlock.
        Backtracking: An jedem Switch werden alle Alternativen ausprobiert, falls Deadlock.
        Die DFS-Laufzeit wird global über einen Controller gesteuert:
        - count: Anzahl der insgesamt besuchten Zellen (max_steps, global für alle DFS-Aufrufe)
        - visited: Alle (pos, direction) Paare, die besucht wurden (cycle prevention, global)
        - seen_agents: Alle Agenten, die auf dem Pfad begegnet wurden (deadlock detection, global)

        Rückgabe:
            steps: Anzahl der insgesamt besuchten Zellen (global, max_steps)
            deadlock_flag: 1, falls Deadlock oder Abbruch, sonst 0
            num_switches: Anzahl der durchlaufenen Switches
            seen_agents: sortierte Liste aller gesehenen Agenten (unique)
            abort_flag: 1, falls Suche wegen max_steps abgebrochen wurde, sonst 0
            target_found_flag: 1, falls Ziel erreicht wurde, sonst 0
        '''
        env = self.env
        distance_map = env.distance_map.get() 

        # Globaler Controller für DFS-Laufzeit und besuchte Knoten/Agenten
        dfs_runtime_controller = {
            'count': 0,           # Gesamtanzahl besuchter Zellen
            'visited': set(),     # Alle besuchten (pos, direction) Paare
            'seen_agents': set()  # Alle gesehenen Agenten
        }
        def dfs(pos, direction, switch_stack, num_switches, backward_trace, max_steps):
            cur_dist = distance_map[handle, pos[0], pos[1], direction] 

            # DFS mit globalem Controller für count, visited, seen_agents
            if dfs_runtime_controller['count'] >= max_steps:
                # Abbruch wegen Schrittbegrenzung
                return cur_dist, 0, num_switches, 1, -1
            if pos == target and not backward_trace:
                # Ziel erreicht
                return cur_dist, 0, num_switches, 0, 1 
            if (pos, direction) in dfs_runtime_controller['visited']:
                # Zyklus erkannt
                return cur_dist, -1, num_switches, 1, -1
            if cur_dist == np.inf and not backward_trace:
                # Kein Pfad zum Ziel
                return -1, -1, num_switches, 1, -1
            
            dfs_runtime_controller['visited'].add((pos, direction))
            dfs_runtime_controller['count'] += 1

            transitions = env.rail.get_transitions(*pos, direction)
            # Deadlock: entgegenkommender Agent
            if self.agent_map is not None: 
                agent_idx = self.agent_map[pos]
                if agent_idx != -1 and agent_idx != handle:
                    if env.agents[agent_idx].direction != direction and not backward_trace:
                        # alle agente werden als potenzielle Deadlock-Quelle betrachtet, 
                        # da sie sich in die Quere kommen könnten (z.B. bei decision_type 3 backward)
                        # nun kann das system aber entscheiden, ich habe einen deadlock und sehe den agenten, der andere mich 
                        # allenfalls nicht, dann ist es ein nicht echter deadlock, 
                        # da er mir noch ausweichen kann 
                        dfs_runtime_controller['seen_agents'].add(agent_idx)
                if agent_idx != -1:
                    if agent_idx != handle:
                        if env.agents[agent_idx].direction != direction:
                            # Deadlock durch entgegenkommenden Agenten - hier geht es momentan nicht mehr weiter!
                            return cur_dist, 1, num_switches, 0, 0
                    else:
                        # Deadlock durch entgegenkommenden Agenten (mich selst) - 
                        # dies kann auftreten falls ich mit descition_type = 4 (bit) 
                        # rückwärts navigiere und auf einen anderen Agenten treffe,
                        # der in die gleiche Richtung unterwegs ist (z.B. beide Agenten 
                        # wollen vor einem Switch warten und haben sich gegenseitig blockiert)
                        return cur_dist, 2, num_switches, 0, 0
                    
            # Switch logic: mehrere Alternativen am Switch, sortiert nach distance_map
            num_trans = fast_count_nonzero(transitions)
            if num_trans > 1:
                alternatives = []
                for ndir in range(4):
                    if transitions[ndir]:
                        npos = get_new_position(pos, ndir)
                        if (npos, ndir) not in dfs_runtime_controller['visited']:
                            dist = distance_map[handle, npos[0], npos[1], ndir]
                            alternatives.append((dist, ndir, npos))
                alternatives.sort(key=lambda x: x[0])
                if not backward_trace:
                    num_switches_changed = 0
                    for _, ndir, npos in alternatives:
                        switch_stack.append((pos, direction, ndir))
                        ret_dfs_dist, ret_deadlock_flag, ret_num_swtich, ret_abort_flag, ret_target_found_flag = dfs(npos, ndir, switch_stack.copy(), num_switches + num_switches_changed, backward_trace, max_steps)
                        if ret_deadlock_flag < 1:
                            return max(ret_dfs_dist, cur_dist), ret_deadlock_flag, ret_num_swtich, ret_abort_flag, ret_target_found_flag
                        num_switches_changed = 1
                        switch_stack.pop()
                else:
                    dfs_dist = 0
                    deadlock_flag = 0
                    num_swtich = 0
                    abort_flag = 0
                    target_found_flag = 0
                    mx_steps = 2.0*max_steps-dfs_runtime_controller['count']/2
                    for _, ndir, npos in alternatives:
                        switch_stack.append((pos, direction, ndir))
                        ret_dfs_dist, ret_deadlock_flag, ret_num_swtich, ret_abort_flag, ret_target_found_flag = dfs(npos, ndir, switch_stack.copy(), num_switches + 1, backward_trace, mx_steps)
                        dfs_dist += ret_dfs_dist
                        deadlock_flag += ret_deadlock_flag
                        num_swtich += ret_num_swtich
                        abort_flag += ret_abort_flag
                        target_found_flag += ret_target_found_flag
                        switch_stack.pop()
                    if (len(alternatives) > 0): 
                        dfs_dist /= len(alternatives)
                        deadlock_flag /= len(alternatives)
                        num_swtich /= len(alternatives)  
                        abort_flag /= len(alternatives)
                        target_found_flag /= len(alternatives)
                    return max(dfs_dist, cur_dist), deadlock_flag, num_swtich, abort_flag, target_found_flag
            else:
                # Normale Fortsetzung: nur eine Richtung möglich
                ndir = fast_argmax(transitions)
                npos = get_new_position(pos, ndir)
                if (npos, ndir) not in dfs_runtime_controller['visited']:
                    ret_dfs_dist, ret_deadlock_flag, ret_num_swtich, ret_abort_flag, ret_target_found_flag = dfs(npos, ndir, switch_stack.copy(), num_switches, backward_trace, max_steps)
                    return max(ret_dfs_dist, cur_dist), ret_deadlock_flag, ret_num_swtich, ret_abort_flag, ret_target_found_flag
            
            # should no occur: keine Alternativen (deadlock) oder Ziel erreicht (handled oben) oder ...
            return cur_dist, 1, num_switches, 0, 0

        dist, deadlock_flag, num_switches, abort_flag, target_found_flag = dfs(start_pos, start_dir, [], 0, backward_trace, max_steps)
        # Nach DFS: Rückgabe der global gesammelten Agenten als sortierte Liste
        seen_agents = sorted(set(dfs_runtime_controller['seen_agents']))
        return dist, deadlock_flag, num_switches, seen_agents, abort_flag, target_found_flag, dfs_runtime_controller['visited']   
    
