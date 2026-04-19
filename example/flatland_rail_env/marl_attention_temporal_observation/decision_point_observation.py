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

        agent_at_switch, agent_near_switch, agent_at_switch_cell, agent_near_switch_cell = \
            self.switchAnalyser.check_agent_decision(position=pos, direction=dir)

        if agent.state.name == "READY_TO_DEPART":
            decision_type = 1.0
        elif agent_at_switch:
            decision_type = 2.0
        elif agent_near_switch_cell:
            decision_type = 3.0
        else:
            decision_type = 0.0

        wait_flag = 0.0
        deadlock_flag = 0.0
        merge_type = 0.0

        if decision_type == 2.0:
            for d in [-1, 0, 1]:
                abs_dir = (dir + d) % 4
                transitions = self.env.rail.get_transitions(*pos, dir)
                if transitions[abs_dir]:
                    npos = get_new_position(pos, abs_dir)
                    if hasattr(self.env, 'agent_map') and self.env.agent_map is not None:
                        agent_on_path = self.env.agent_map[npos] != -1 and self.env.agent_map[npos] != handle
                        if agent_on_path:
                            other_agent = self.env.agents[self.env.agent_map[npos]]
                            if other_agent.state == TrainState.STOPPED or (other_agent.direction + 2) % 4 == abs_dir:
                                deadlock_flag = 1.0
        elif decision_type == 3.0:
            if hasattr(self.env, 'agent_map') and self.env.agent_map is not None:
                for d in range(4):
                    npos = get_new_position(pos, d)
                    if self.env.agent_map[npos] != -1 and self.env.agent_map[npos] != handle:
                        other_agent = self.env.agents[self.env.agent_map[npos]]
                        if (other_agent.direction + 2) % 4 == dir:
                            merge_type = 2.0
                            wait_flag = 1.0
                            break
                        elif other_agent.direction == dir:
                            merge_type = 1.0
                            wait_flag = 1.0

        features = np.zeros(self.feature_len, dtype=np.float32)
        features[0] = decision_type
        features[1] = float(agent.state.value)
        features[2] = float(dir)
        features[3] = float(pos[0])
        features[4] = float(pos[1])
        features[5] = float(target[0])
        features[6] = float(target[1])
        features[7] = float(agent_at_switch)
        features[8] = float(agent_near_switch_cell)
        features[9] = float(agent_at_switch_cell)
        features[10] = float(agent_near_switch)
        can_cross = 0.0
        if agent_near_switch_cell:
            if hasattr(self.env, 'agent_map') and self.env.agent_map is not None:
                for d in range(4):
                    npos = get_new_position(pos, d)
                    if self.env.agent_map[npos] != -1 and self.env.agent_map[npos] != handle:
                        other_agent = self.env.agents[self.env.agent_map[npos]]
                        if (other_agent.direction + 2) % 4 == dir:
                            can_cross = 1.0
                            break
        features[11] = can_cross
        features[12] = wait_flag
        features[13] = deadlock_flag
        features[14] = merge_type
        # Rest bleibt 0
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
