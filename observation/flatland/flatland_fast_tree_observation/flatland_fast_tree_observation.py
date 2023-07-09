from typing import Optional, List, Union

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
        self.target_found = 0

    def clear(self):
        self.visited = []
        self.switches = []
        self.opp_direction_agents = []
        self.same_direction_agents = []
        self.final_pos = None
        self.final_dir = None
        self.target_found = 0

    def callback(self, handle, agent, position, direction, action, possible_transitions) -> bool:
        self.visited.append(position)
        if self.final_pos is None:
            self.final_pos = position
            self.final_dir = direction
        self.target_found = fast_position_equal(agent.target, position)
        if self.target_found == 1:
            self.final_pos = position
            self.final_dir = direction
            return False

        if fast_count_nonzero(possible_transitions) > 1:
            self.switches.append(position)

        i_agent = self.env.agent_positions[position]
        if i_agent != handle:
            if direction != agent.direction:
                self.opp_direction_agents.append(i_agent)
                return False
            else:
                self.same_direction_agents.append(i_agent)
        self.final_pos = position
        self.final_dir = direction
        return True


class WalkToNextDecisionPoint(ShortestDistanceWalker):
    def __init__(self, env: RailEnv):
        super(WalkToNextDecisionPoint, self).__init__(env)
        self.visited = []
        self.final_pos = None
        self.final_dir = None
        self.target_found = 0

    def clear(self):
        self.visited = []
        self.final_pos = None
        self.final_dir = None
        self.target_found = 0

    def callback(self, handle, agent, position, direction, action, possible_transitions) -> bool:
        if self.final_pos is None:
            self.final_pos = position
            self.final_dir = direction
        self.target_found = int(fast_position_equal(agent.target, position))
        if self.target_found == 1:
            return False

        self.visited.append(position)
        if fast_count_nonzero(possible_transitions) > 1:
            return False

        possible_transitions_opp_dir = self.env.rail.get_transitions(*position, (direction + 2 % 4))
        if fast_count_nonzero(possible_transitions_opp_dir) > 1:
            return False

        self.final_pos = position
        self.final_dir = direction
        return True


class FlatlandFastTreeObservation(ObservationBuilder):

    def __init__(self):
        super(FlatlandFastTreeObservation, self).__init__()
        self.observation_dim = OBSERVATION_DIM
        self.previous_observations = {}
        self.agents_path_maps_cache = {}
        self.walk_to_next_decision_point: Union[WalkToNextDecisionPoint, None] = None
        self.discoverer_shortest_distance_walker: Union[DiscovererShortestDistanceWalker, None] = None

    def reset(self):
        self.previous_observations = {}
        self.agents_path_maps_cache = {}

    def set_env(self, env: RailEnv):
        super(FlatlandFastTreeObservation, self).set_env(env)
        if self.walk_to_next_decision_point is None:
            self.walk_to_next_decision_point = WalkToNextDecisionPoint(self.env)
        else:
            self.walk_to_next_decision_point.reset(env)
        if self.discoverer_shortest_distance_walker is None:
            self.discoverer_shortest_distance_walker = DiscovererShortestDistanceWalker(self.env)
        else:
            self.discoverer_shortest_distance_walker.reset(env)

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

        self.walk_to_next_decision_point.clear()
        self.discoverer_shortest_distance_walker.clear()

        agent = self.env.agents[handle]
        distance_map = self.env.distance_map.get()

        agent_pos, agent_dir, agent_state = \
            FlatlandFastTreeObservation.get_agent_position_and_direction(agent)
        visited.append(agent_pos)

        # update state / observation
        # one-hot
        observation[0] = int(agent_state == 0)
        observation[1] = int(agent_state == 1)
        observation[2] = int(agent_state == 2)
        observation[3] = int(agent_state == 3)
        observation[4] = int(agent_state == 4)
        observation[5] = int(agent_state == 5)
        observation[6] = int(agent_state == 6)

        self.walk_to_next_decision_point.walk_to_target(handle, agent_pos, agent_dir, 50)
        observation[7] = len(self.walk_to_next_decision_point.visited)

        visited = visited + self.walk_to_next_decision_point.visited
        if self.walk_to_next_decision_point.final_pos is not None:
            agent_pos = self.walk_to_next_decision_point.final_pos
            agent_dir = self.walk_to_next_decision_point.final_dir

        if not self.walk_to_next_decision_point.target_found:
            current_cell_dist = distance_map[handle, agent_pos[0], agent_pos[1], agent_dir]

            orientation = agent_dir
            possible_transitions = self.env.rail.get_transitions(*agent_pos, agent_dir)
            if fast_count_nonzero(possible_transitions) == 1:
                orientation = fast_argmax(possible_transitions)

            for dir_loop, branch_direction in enumerate([(orientation + dir_loop) % 4 for dir_loop in range(-1, 3)]):
                if possible_transitions[branch_direction]:
                    new_position = get_new_position(agent_pos, branch_direction)
                    if new_position is not None:
                        new_cell_dist = distance_map[handle, new_position[0], new_position[1], branch_direction]
                        if not np.isinf(new_cell_dist):
                            observation[8 + branch_direction] = int(current_cell_dist > new_cell_dist)
                        visited.append(new_position)

                    self.discoverer_shortest_distance_walker.walk_to_target(handle, new_position, branch_direction, 50)
                    if self.discoverer_shortest_distance_walker.final_pos is not None:
                        final_cell_dist = distance_map[
                            handle,
                            self.discoverer_shortest_distance_walker.final_pos[0],
                            self.discoverer_shortest_distance_walker.final_pos[1],
                            self.discoverer_shortest_distance_walker.final_dir]
                        if not np.isinf(final_cell_dist):
                            observation[12 + branch_direction] = int(current_cell_dist > final_cell_dist)
                        observation[16 + branch_direction] = \
                            len(self.discoverer_shortest_distance_walker.switches)
                        observation[20 + branch_direction] = \
                            len(self.discoverer_shortest_distance_walker.same_direction_agents)
                        observation[24 + branch_direction] = \
                            int(self.discoverer_shortest_distance_walker.target_found)

                    visited = visited + self.discoverer_shortest_distance_walker.visited
        self.env.dev_obs_dict.update({handle: visited})

        observation[np.isinf(observation)] = -1
        observation[np.isnan(observation)] = -1

        return observation
