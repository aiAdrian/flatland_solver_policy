from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.step_utils.states import TrainState
from flatland.core.grid.grid4_utils import get_new_position
import numpy as np
from flatland_railway_extension.RailroadSwitchAnalyser import RailroadSwitchAnalyser
from marl_attention_temporal_observation.decision_point_observation import DecisionPointObservation
from collections import namedtuple
from typing import Optional, List, Any

class ExperimentalObservation(ObservationBuilder):
    """
    Base observation (30D self + 7D multi-agent features)
    """
    def __init__(self,
                 max_path_length: int = 60,
                 lookahead_cost_limit: int = 40,
                 max_agent_dist: int = 10):
        super().__init__()
        self.max_path_length = max_path_length
        self.lookahead_cost_limit = lookahead_cost_limit
        self.max_agent_dist = max_agent_dist
        self.env = None
        self.agent_map = None
        self.switchAnalyser: RailroadSwitchAnalyser  = None
        self.walker = None
        self.observation_space = np.zeros(
            ExperimentalObservation.getObservationSize(),
            dtype=np.float32
        )
        print(">> ExperimentalObservation loaded.")

    @staticmethod
    def getObservationSize() -> int:
        return 30

    def set_env(self, env):
        """Set environment and initialize components."""
        self.env = env
        if self.switchAnalyser is None:
            self.switchAnalyser = RailroadSwitchAnalyser(env)
        from .walk_to_next_decision_point import WalkToNextDecisionPoint
        if self.walker is None:
            self.walker = WalkToNextDecisionPoint(env)

    def reset(self):
        self.switchAnalyser = RailroadSwitchAnalyser(self.env)
        from .walk_to_next_decision_point import WalkToNextDecisionPoint
        self.walker = WalkToNextDecisionPoint(self.env)

    @staticmethod
    def get_pos_dir(agent: EnvAgent):
        pos = agent.position if agent.position is not None else agent.initial_position
        dir = agent.direction if agent.direction is not None else agent.initial_direction
        return pos, dir


    @staticmethod
    def get_decision_point_observation(env, handle, switchAnalyser, walker, max_path_length, lookahead_cost_limit, max_agent_dist):
        agent = env.agents[handle]
        pos = agent.position if agent.position is not None else agent.initial_position
        direction = agent.direction if agent.direction is not None else agent.initial_direction
        if pos is None or direction is None or not agent.state.is_on_map_state():
            # Agent ist nicht auf der Map
            return np.zeros(30, dtype=np.float32), []

        # Beispielhafte Feature-Berechnung (hier kannst du deine Logik anpassen) 
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

    def get(self, handle: int):
        agent = self.env.agents[handle]
        pos, direction = self.get_pos_dir(agent)
        if pos is None or direction is None or not agent.state.is_on_map_state():
            # Agent ist nicht auf der Map
            return np.zeros(ExperimentalObservation.getObservationSize(), dtype=np.float32), []

        # Analyse der Entscheidungsstellen und Pfade
        decision_obs, opp_agents = ExperimentalObservation.get_decision_point_observation(
            self.env, handle, self.switchAnalyser, self.walker, self.max_path_length, self.lookahead_cost_limit, self.max_agent_dist
        )
        return decision_obs, opp_agents

    def get_many(self, handles: Optional[List[int]] = None) -> Any:
        """Return list of (obs, opp_agents) tuples for compatibility with TemporalMultiAgentObservation."""
        h, w = self.env.height, self.env.width
        self.agent_map = np.full((h, w), -1, dtype=int)
        for agent in self.env.agents:
            pos, _ = self.get_pos_dir(agent)
            if pos is None:
                continue
            if agent.state.is_on_map_state():
                self.agent_map[pos] = agent.handle
        self.walker.clear(self.agent_map)
        
        if handles is None:
            handles = list(range(len(self.env.agents)))
        
        # Return list of (obs, opp_agents) 2-tuples for TemporalMultiAgentObservation compatibility
        all_obs = [self.get(handle) for handle in handles]
        return all_obs
