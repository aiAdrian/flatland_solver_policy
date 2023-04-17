import numpy as np
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_count_nonzero, fast_argmax
from flatland.envs.rail_env import RailEnv, RailEnvActions


class ShortestDistanceWalker:
    def __init__(self, env: RailEnv):
        self.env = env

    def walk(self, handle, position, direction):
        possible_transitions = self.env.rail.get_transitions(*position, direction)
        num_transitions = fast_count_nonzero(possible_transitions)
        if num_transitions == 1:
            new_direction = fast_argmax(possible_transitions)
            new_position = get_new_position(position, new_direction)

            dist = self.env.distance_map.get()[handle, new_position[0], new_position[1], new_direction]
            return new_position, new_direction, dist, RailEnvActions.MOVE_FORWARD, possible_transitions
        else:
            min_distances = []
            positions = []
            directions = []
            for new_direction in [(direction + i) % 4 for i in range(-1, 2)]:
                if possible_transitions[new_direction]:
                    new_position = get_new_position(position, new_direction)
                    min_distances.append(
                        self.env.distance_map.get()[handle, new_position[0], new_position[1], new_direction])
                    positions.append(new_position)
                    directions.append(new_direction)
                else:
                    min_distances.append(np.inf)
                    positions.append(None)
                    directions.append(None)

        a = self.get_action(handle, min_distances)
        return positions[a], directions[a], min_distances[a], a + 1, possible_transitions

    def get_action(self, handle, min_distances):
        return np.argmin(min_distances)

    def callback(self, handle, agent, position, direction, action, possible_transitions):
        pass

    def get_agent_position_and_direction(self, handle):
        agent = self.env.agents[handle]
        if agent.position is not None:
            position = agent.position
        else:
            position = agent.initial_position
        direction = agent.direction
        return position, direction

    def walk_to_target(self, handle, position=None, direction=None, max_step=500):
        if position is None and direction is None:
            position, direction = self.get_agent_position_and_direction(handle)
        elif position is None:
            position, _ = self.get_agent_position_and_direction(handle)
        elif direction is None:
            _, direction = self.get_agent_position_and_direction(handle)

        agent = self.env.agents[handle]
        step = 0
        while (position != agent.target) and (step < max_step):
            position, direction, dist, action, possible_transitions = self.walk(handle, position, direction)
            if position is None:
                break
            self.callback(handle, agent, position, direction, action, possible_transitions)
            step += 1

    def callback_one_step(self, handle, agent, position, direction, action, possible_transitions):
        pass

    def walk_one_step(self, handle):
        agent = self.env.agents[handle]
        if agent.position is not None:
            position = agent.position
        else:
            position = agent.initial_position
        direction = agent.direction
        possible_transitions = (0, 1, 0, 0)
        if (position != agent.target):
            new_position, new_direction, dist, action, possible_transitions = self.walk(handle, position, direction)
            if new_position is None:
                return position, direction, RailEnvActions.STOP_MOVING, possible_transitions
            self.callback_one_step(handle, agent, new_position, new_direction, action, possible_transitions)
        return new_position, new_direction, action, possible_transitions
