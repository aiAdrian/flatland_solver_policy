from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.step_utils.states import TrainState
from flatland.core.grid.grid4_utils import get_new_position
import numpy as np
from flatland_railway_extension.RailroadSwitchAnalyser import RailroadSwitchAnalyser
from .decision_point_observation import DecisionPointObservation
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
            ExperimentalObservation.getObservationSize() - ExperimentalObservation.getObservationOthersExtraSize(),
            dtype=np.float32
        )

    @staticmethod
    def getObservationOthersExtraSize() -> int:
        return 7

    @staticmethod
    def getObservationSize() -> int:
        return 5 + 3 * 39 + ExperimentalObservation.getObservationOthersExtraSize()

    def reset(self):
        self.switchAnalyser = RailroadSwitchAnalyser(self.env)
        from .walk_to_next_decision_point import WalkToNextDecisionPoint
        self.walker = WalkToNextDecisionPoint(self.env)

    @staticmethod
    def get_pos_dir(agent: EnvAgent):
        pos = agent.position if agent.position is not None else agent.initial_position
        dir = agent.direction if agent.direction is not None else agent.initial_direction
        return pos, dir

    def get_many(self, handles: Optional[List[int]] = None) -> Any:
        h, w = self.env.height, self.env.width
        self.agent_map = np.full((h, w), -1, dtype=int)
        for agent in self.env.agents:
            pos, _ = self.get_pos_dir(agent)
            if pos is None:
                continue
            if agent.state.is_on_map_state():
                self.agent_map[pos] = agent.handle
        self.walker.clear(self.agent_map)
        all_obs = super().get_many(handles)
        states = []
        for obs_agent_handle in range(len(all_obs)):
            obs_self, opp_agents = all_obs[obs_agent_handle]
            obs_self = obs_self.copy()
            for d in range(7):
                obs_self = np.append(obs_self, 0)
            other_list = []
            for my_opp_agent in opp_agents:
                if my_opp_agent.handle != obs_agent_handle:
                    obs_other, _ = all_obs[my_opp_agent.handle]
                    obs_other = obs_other.copy()
                    obs_other = np.append(obs_other, my_opp_agent.same_direction)
                    self_is_also_in_opp_agents = [0, 0, 0]
                    self_is_also_in_opp_agents_dir = [0, 0, 0]
                    self_is_also_in_opp_agents[my_opp_agent.detected_branch_action] = 1
                    self_is_also_in_opp_agents_dir[my_opp_agent.detected_branch_action] = my_opp_agent.opp_direction
                    for d in self_is_also_in_opp_agents:
                        obs_other = np.append(obs_other, d)
                    for d in self_is_also_in_opp_agents_dir:
                        obs_other = np.append(obs_other, d)
                    other_list.append(obs_other)
            state = (obs_self, other_list)
            states.append(state)
        return states
    # ...restliche Methoden (wie _analyze_direction_branches, get) bitte analog auslagern...
