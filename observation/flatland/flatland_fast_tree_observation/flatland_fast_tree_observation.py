from typing import Optional, List

import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.fast_methods import fast_count_nonzero, fast_argmax, fast_position_equal
from flatland.envs.rail_env import RailEnv

from utils.flatland.shortest_distance_walker import ShortestDistanceWalker

OBSERVATION_DIM = 28


class DiscovererShortestDistanceWalker(ShortestDistanceWalker):
    def __init__(self, env: RailEnv):
        super(DiscovererShortestDistanceWalker, self).__init__(env)
        self.visited = []
        self.switches = []
        self.opp_direction_agents = []
        self.same_direction_agents = []
        self.final_pos = None
        self.final_dir = None
        self.target_found = None

    def callback(self, handle, agent, position, direction, action, possible_transitions) -> bool:
        self.visited.append(position)
        self.target_found = fast_position_equal(agent.target, position)
        if fast_count_nonzero(possible_transitions) > 1:
            self.switches.append(position)

        for i_agent, agent in enumerate(self.env.agents):
            if fast_position_equal(agent.position, position):
                if direction != agent.direction:
                    self.opp_direction_agents.append(i_agent)
                    return False
                else:
                    self.same_direction_agents.append(i_agent)
        self.final_pos = position
        self.final_dir = direction
        return True


class FlatlandFastTreeObservation(ObservationBuilder):

    def __init__(self):
        super(FlatlandFastTreeObservation, self).__init__()
        self.observation_dim = OBSERVATION_DIM
        self.previous_observations = {}
        self.agents_path_maps_cache = {}

    def reset(self):
        self.previous_observations = {}
        self.agents_path_maps_cache = {}

    @staticmethod
    def get_agent_position_and_direction(agent: EnvAgent):
        position = agent.position if agent.position is not None else agent.initial_position
        direction = agent.direction if agent.direction is not None else agent.initial_direction
        return position, direction, agent.state

    def get_many(self, handles: Optional[List[int]] = None):
        observations = super().get_many(handles)
        return observations

    def get(self, handle: int = 0):
        observation = np.zeros(self.observation_dim) - 1

        visited = []

        agent = self.env.agents[handle]
        distance_map = self.env.distance_map.get()

        agent_pos, agent_dir, agent_state = \
            FlatlandFastTreeObservation.get_agent_position_and_direction(agent)

        # update state / observation
        observation[0] = int(agent_state == 0)
        observation[1] = int(agent_state == 1)
        observation[2] = int(agent_state == 2)
        observation[3] = int(agent_state == 3)
        observation[4] = int(agent_state == 4)
        observation[5] = int(agent_state == 5)
        observation[6] = int(agent_state == 6)

        current_cell_dist = distance_map[handle, agent_pos[0], agent_pos[1], agent_dir]
        possible_transitions = self.env.rail.get_transitions(*agent_pos, agent_dir)
        orientation = agent_dir
        if fast_count_nonzero(possible_transitions) == 1:
            orientation = fast_argmax(possible_transitions)

        visited.append(agent_pos)

        for dir_loop, branch_direction in enumerate([(orientation + dir_loop) % 4 for dir_loop in range(-1, 3)]):

            if possible_transitions[branch_direction]:
                new_position = get_new_position(agent_pos, branch_direction)
                if new_position is not None:
                    new_cell_dist = distance_map[handle, new_position[0], new_position[1], branch_direction]
                    if not np.isinf(new_cell_dist):
                        observation[8 + branch_direction] = int(current_cell_dist > new_cell_dist)
                    visited.append(new_position)
                discoverer = DiscovererShortestDistanceWalker(self.env)
                discoverer.walk_to_target(handle, new_position, branch_direction, 50)
                if discoverer.final_pos is not None:
                    final_cell_dist = distance_map[
                        handle, discoverer.final_pos[0], discoverer.final_pos[1], discoverer.final_dir]
                    if not np.isinf(final_cell_dist):
                        observation[12 + branch_direction] = int(current_cell_dist > final_cell_dist)
                    observation[16 + branch_direction] = len(discoverer.switches)
                    observation[20 + branch_direction] = len(discoverer.same_direction_agents)
                    observation[24 + branch_direction] = int(discoverer.target_found)

                visited = visited + discoverer.visited
        self.env.dev_obs_dict.update({handle: visited})

        observation[np.isinf(observation)] = -1
        observation[np.isnan(observation)] = -1

        return observation
