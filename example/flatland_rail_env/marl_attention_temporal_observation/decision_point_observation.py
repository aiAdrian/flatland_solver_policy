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
        Backtracking: An jedem Switch werden alle Alternativen ausprobiert, falls Deadlock.
        Gibt (steps, deadlock_flag, num_switches, seen_agents, abort_flag, target_found_flag) zurück.
        '''
        env = self.env
        distance_map = env.distance_map.get()
        agent_map = env.agent_map if hasattr(env, 'agent_map') else None

        dfs_runtime_controller = {
            'count': 0,
            'visited': set(),
            'seen_agents': set()
        }
        def dfs(pos, direction, switch_stack, num_switches):
            if dfs_runtime_controller['count'] >= max_steps:
                return dfs_runtime_controller['count'], 1, num_switches, 1, 0
            if pos == target:
                return dfs_runtime_controller['count'], 0, num_switches, 0, 1
            if not self._is_in_grid(pos):
                return dfs_runtime_controller['count'], 1, num_switches, 0, 0
            if (pos, direction) in dfs_runtime_controller['visited']:
                return dfs_runtime_controller['count'], 1, num_switches, 0, 0
            dfs_runtime_controller['visited'].add((pos, direction))
            dfs_runtime_controller['count'] += 1

            transitions = env.rail.get_transitions(*pos, direction)
            
            # Deadlock: entgegenkommender Agent
            if agent_map is not None:
                agent_idx = agent_map[pos]
                if agent_idx != -1 and agent_idx != handle:
                    dfs_runtime_controller['seen_agents'].add(agent_idx)
                if agent_idx != -1 and env.agents[agent_idx].direction == (direction + 2) % 4:
                    return dfs_runtime_controller['count'], 1, num_switches, 0, 0
                
            # Switch logic: multiple transitions = switch
            num_trans = fast_count_nonzero(transitions)
            if num_trans > 1:
                alternatives = []
                for i in range(4):
                    if transitions[i]:
                        npos = get_new_position(pos, i)
                        if (npos, i) not in dfs_runtime_controller['visited']:
                            dist = distance_map[handle, npos[0], npos[1], i]
                            alternatives.append((dist, i, npos))
                alternatives.sort(key=lambda x: x[0])
                for _, i, npos in alternatives:
                    switch_stack.append((pos, direction, i))
                    res = dfs(npos, i, switch_stack.copy(), num_switches+1)
                    if res[1] == 0:
                        return res
                    switch_stack.pop()
                return dfs_runtime_controller['count'], 1, num_switches, 0, 0
            
            # Normal continuation: only one direction
            i = fast_argmax(transitions)
            npos = get_new_position(pos, i)
            if (npos, i) not in dfs_runtime_controller['visited']:
                return dfs(npos, i, switch_stack.copy(), num_switches)
            # Dead end
            return dfs_runtime_controller['count'], 1, num_switches, 0, 0

        steps, deadlock_flag, num_switches, abort_flag, target_found_flag = dfs(start_pos, start_dir, [], 0)
        seen_agents = sorted(set(dfs_runtime_controller['seen_agents']))
        return steps, deadlock_flag, num_switches, seen_agents, abort_flag, target_found_flag
    
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
        self.feature_len = 33 
        print(">> DecisionPointObservation loaded.")

    def set_env(self, env):
        self.env = env
        self.switchAnalyser = None

    def reset(self):
        self.switchAnalyser = None

    @staticmethod
    def getObservationSize() -> int:
        return 31

    def get(self, handle: int = 0):
        # init features
        features = np.zeros(self.feature_len, dtype=np.float32)

        # Get base information about the agent and its environment
        agent = self.env.agents[handle]  
        pos = agent.position if agent.position is not None else agent.initial_position
        dir = agent.direction if agent.direction is not None else agent.initial_direction
        target = agent.target
        if pos is None or target is None:
            return (features-1, [])

        # retrieve distance map for pathfinding features
        distance_map = self.env.distance_map.get()
        curr_dist = self.env.distance_map.get()[handle, pos[0], pos[1], dir] if self._is_in_grid(pos) else -1

        # classify decision point type based on agent state and position and infrastructure / cell 
        if self.switchAnalyser is None:
            from flatland_railway_extension.RailroadSwitchAnalyser import RailroadSwitchAnalyser
            self.switchAnalyser = RailroadSwitchAnalyser(self.env)

        agent_at_switch, agent_near_switch, switch_cell, near_switch_cell = \
            self.switchAnalyser.check_agent_decision(position=pos, direction=dir)


        # Check if there's an agent at the current position (e.g., for deadlock detection)
        agent_idx = self.env.agent_map[pos] if hasattr(self.env, 'agent_map') and self.env.agent_map is not None else -1
        if agent_idx != -1:
            pass

        # classify decision point type
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
        


        # Feature 0: decision_type
        features[0] = decision_type
 
        transitions = self.env.rail.get_transitions(*pos, dir)
        best_hint = [0.0, 0.0, 0.0]  # [l, f, r]
        delta_dist_fwd = 0
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
                    if idx == 1:  # forward
                        delta_dist_fwd = dist - curr_dist if curr_dist != -1 and dist != np.inf else 0
            if best_idx != -1:
                best_hint[best_idx] = 1.0
        features[1:4] = best_hint
 

        opp_agents = set()
        abort_flag = 0
        # decision_type 2: agent on a switch -> for each direction [dist, deadlock, switches]
        if decision_type == 2.0:
            rel_dirs = [(-1) % 4, 0, 1, 2]  # left, forward, right, reverse
            for i, rel_dir in enumerate(rel_dirs):
                abs_dir = (dir + rel_dir) % 4
                base = 4 + i*4
                if transitions[abs_dir]:
                    npos = get_new_position(pos, abs_dir)
                    dist, deadlock, switches, seen, abort, target_found = self._navigate_direction(handle, npos, abs_dir, target)
                    opp_agents.update(seen)
                    abort_flag = max(abort_flag, abort) 
                else:
                    dist, deadlock, switches, abort, target_found  = -1, -1, -1, -1, -1
                features[base] = dist         # 4,8,12,16: dist
                features[base + 1] = deadlock   # 5,9,13,17: deadlock
                features[base + 2] = switches   # 6,10,14,18: switches
                features[base + 3] = dist - curr_dist if dist != -1 and curr_dist != -1 and dist != np.inf else -1
                features[base + 4] = target_found  # 8,12,16,20: target_found_flag
                features[base + 5] = abort  # 9,13,17,21: abort_flag

        # decision_type 3: agent near a merging switch 
        # -  look forward and backward for crossing and ordering (priority) logic 
        #    (e.g., if forward path is blocked by another agent, can we merge/cross? 
        #    If backward path is blocked, do we have priority?)
        # - cannot branch, but can merge/cross) -> for each direction 
        #   [dist, deadlock, switches, delta_dist, target_found, abort]
        if decision_type == 3.0: 
            # Forward: 1 step forward (merge/crossing logic)
            forward_dir = fast_argmax(transitions)
            new_position = get_new_position(pos, dir)

            npos_fwd = new_position
            if npos_fwd is not None and self._is_in_grid(npos_fwd):
                dist_fwd, deadlock_fwd, switches_fwd, seen_fwd, abort_fwd, target_found_fwd = self._navigate_direction(handle, npos_fwd, forward_dir, target)
                opp_agents.update(seen_fwd)
                abort_flag = max(abort_flag, abort_fwd) 
                dist_fwd_map = self.env.distance_map.get()[handle, npos_fwd[0], npos_fwd[1], forward_dir]
                delta_dist_fwd = dist_fwd_map - curr_dist if curr_dist != -1 and dist_fwd_map != np.inf else -1
            else:
                dist_fwd, deadlock_fwd, switches_fwd, abort_fwd, target_found_fwd = -1, -1, -1, -1, -1
                delta_dist_fwd = -1

            features[21] = dist_fwd          # 20: dist_fwd
            features[22] = deadlock_fwd      # 21: deadlock_fwd
            features[23] = switches_fwd      # 22: switches_fwd
            features[24] = delta_dist_fwd    # 23: delta_dist_fwd
            features[25] = target_found_fwd  # 24: target_found_fwd
            features[26] = abort_fwd         # 25: abort_flag_fwd

            # Backward: 1 step backward (merge/crossing logic) 
            new_position = get_new_position(pos, dir)
            reverse_dir = (forward_dir + 2) % 4
            npos_bwd = get_new_position(new_position, reverse_dir)
            if npos_bwd is not None and self._is_in_grid(npos_bwd):
                dist_bwd, deadlock_bwd, switches_bwd, seen_bwd, abort_bwd, target_found_bwd = self._navigate_direction(handle, npos_bwd, reverse_dir, target)
                opp_agents.update(seen_bwd) 
                dist_bwd_map = self.env.distance_map.get()[handle, npos_bwd[0], npos_bwd[1], reverse_dir]
                delta_dist_bwd = dist_bwd_map - curr_dist if curr_dist != -1 and dist_bwd_map != np.inf else -1
            else:
                dist_bwd, deadlock_bwd, switches_bwd, abort_bwd, target_found_bwd  = -1, -1, -1, -1, -1
                delta_dist_bwd = -1
            features[27] = dist_bwd          # 24: dist_bwd
            features[28] = deadlock_bwd      # 25: deadlock_bwd
            features[29] = switches_bwd      # 26: switches_bwd
            features[30] = delta_dist_bwd    # 27: delta_dist_bwd
            features[31] = target_found_bwd  # 28: target_found_bwd
            features[32] = abort_bwd         # 29: abort_flag_bwd
         


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
