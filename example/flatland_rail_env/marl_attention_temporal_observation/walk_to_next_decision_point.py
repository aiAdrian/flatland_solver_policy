from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.rail_env import RailEnv
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.step_utils.states import TrainState
from utils.flatland.shortest_distance_walker import ShortestDistanceWalker
import numpy as np

class WalkToNextDecisionPoint(ShortestDistanceWalker):
    def __init__(self, env: RailEnv):
        super(WalkToNextDecisionPoint, self).__init__(env)
        self.clear(None)
        self.switchAnalyser = None
        self.nbr_join_switch_visited = 0
        self.stop_after_max_join_switch = np.inf

    def clear(self, agent_map):
        self.agent_map = agent_map
        self.visited = []
        self.path = []
        self.switch = []
        self.dir_switch = []
        self.near_to_switch = []
        self.near_to_dir_switch = []
        self.other_agent_handles = []
        self.dir_at_agent_handles = []
        self.final_pos = None
        self.final_dir = None
        self.target_found = 0
        self.nbr_join_switch_visited = 0

    def set_max_join_switch(self, max_value: int):
        self.stop_after_max_join_switch = max_value

    def callback(self, handle, agent, position, direction, action, possible_transitions) -> bool:
        if self.final_pos is None:
            self.final_pos = position
            self.final_dir = direction
        self.target_found = int(np.array_equal(agent.target, position))
        if self.target_found == 1:
            return False

        # Hier kann ggf. switchAnalyser verwendet werden, falls benötigt
        self.visited.append((position, direction))
        self.path.append(position)
        self.final_pos = position
        self.final_dir = direction
        return True
