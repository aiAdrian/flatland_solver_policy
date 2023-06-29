from functools import lru_cache
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from flatland.envs.fast_methods import fast_count_nonzero
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.step_utils.states import TrainState

from environment.environment import Environment
from policy.heuristic_policy.heuristic_policy import HeuristicPolicy
from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.shortest_distance_walker \
    import ShortestDistanceWalker

# activate LRU caching
flatland_deadlock_avoidance_policy_lru_cache_functions = []


def _enable_flatland_deadlock_avoidance_policy_lru_cache(*args, **kwargs):
    def decorator(func):
        func = lru_cache(*args, **kwargs)(func)
        flatland_deadlock_avoidance_policy_lru_cache_functions.append(func)
        return func

    return decorator


def _send_flatland_deadlock_avoidance_policy_data_change_signal_to_reset_lru_cache():
    for func in flatland_deadlock_avoidance_policy_lru_cache_functions:
        func.cache_clear()


class DeadlockAvoidanceShortestDistanceWalker(ShortestDistanceWalker):
    def __init__(self, env: RailEnv):
        super().__init__(env)
        self.shortest_distance_agent_map = None
        self.full_shortest_distance_agent_map = None
        self.agent_positions = None
        self.opp_agent_map = {}
        self.same_agent_map = {}

    def reset(self, env: RailEnv):
        super(DeadlockAvoidanceShortestDistanceWalker, self).reset(env)
        self.shortest_distance_agent_map = None
        self.full_shortest_distance_agent_map = None
        self.agent_positions = None
        self.opp_agent_map = {}
        self.same_agent_map = {}
        _send_flatland_deadlock_avoidance_policy_data_change_signal_to_reset_lru_cache()

    def clear(self, agent_positions):
        self.shortest_distance_agent_map = np.zeros((self.env.get_num_agents(),
                                                     self.env.height,
                                                     self.env.width),
                                                    dtype=int) - 1

        self.full_shortest_distance_agent_map = np.zeros((self.env.get_num_agents(),
                                                          self.env.height,
                                                          self.env.width),
                                                         dtype=int) - 1

        self.agent_positions = agent_positions

        self.opp_agent_map = {}
        self.same_agent_map = {}

    def getData(self):
        return self.shortest_distance_agent_map, self.full_shortest_distance_agent_map

    def callback(self, handle, agent, position, direction, action, possible_transitions):
        opp_a = self.agent_positions[position]
        if opp_a != -1 and opp_a != handle:
            if self.env.agents[opp_a].direction != direction:
                d = self.opp_agent_map.get(handle, [])
                if opp_a not in d:
                    d.append(opp_a)
                self.opp_agent_map.update({handle: d})
            else:
                if len(self.opp_agent_map.get(handle, [])) == 0:
                    d = self.same_agent_map.get(handle, [])
                    if opp_a not in d:
                        d.append(opp_a)
                    self.same_agent_map.update({handle: d})

        if len(self.opp_agent_map.get(handle, [])) == 0:
            if self._is_no_switch_cell(position):
                self.shortest_distance_agent_map[(handle, position[0], position[1])] = 1
        self.full_shortest_distance_agent_map[(handle, position[0], position[1])] = 1

    @_enable_flatland_deadlock_avoidance_policy_lru_cache()
    def _is_no_switch_cell(self, position) -> bool:
        for new_dir in range(4):
            possible_transitions = self.env.rail.get_transitions(*position, new_dir)
            num_transitions = fast_count_nonzero(possible_transitions)
            if num_transitions > 1:
                return False
        return True


# define Python user-defined exceptions
class InvalidRawEnvironmentException(Exception):
    def __init__(self, env, message="This policy works only with a RailEnv or its specialized version. "
                                    "Please check the raw_env . "):
        self.env = env
        self.message = message
        super().__init__(self.message)


class DeadLockAvoidancePolicy(HeuristicPolicy):
    def __init__(self, env: RailEnv,
                 action_size: int,
                 min_free_cell=1,
                 enable_eps=False,
                 show_debug_plot=False):
        super(HeuristicPolicy, self).__init__()
        self.env: RailEnv = env
        self.loss = 0
        self.action_size = action_size
        self.agent_can_move = {}
        self.show_debug_plot = show_debug_plot
        self.enable_eps = enable_eps
        self.shortest_distance_walker: Union[DeadlockAvoidanceShortestDistanceWalker, None] = None
        self.min_free_cell = min_free_cell
        self.agent_positions = None

    def get_name(self):
        return self.__class__.__name__

    def step(self, handle, state, action, reward, next_state, done):
        pass

    def act(self, handle, state, eps=0.):
        # Epsilon-greedy action selection
        if self.enable_eps:
            if np.random.random() < eps:
                return np.random.choice(np.arange(self.action_size))

        # agent = self.env.agents[state[0]]
        check = self.agent_can_move.get(handle, None)
        act = RailEnvActions.STOP_MOVING
        if check is not None:
            act = check[3]
        return act

    def reset(self, env: Environment):
        self.env = env.get_raw_env()
        if self.shortest_distance_walker is not None:
            self.shortest_distance_walker.reset(self.env)
        self.shortest_distance_walker = None
        self.agent_positions = None
        self.shortest_distance_walker = None

    def start_step(self, train):
        self._build_agent_position_map()
        self._shortest_distance_mapper()
        self._extract_agent_can_move()

    def _build_agent_position_map(self):
        # build map with agent positions (only active agents)
        self.agent_positions = np.zeros((self.env.height, self.env.width), dtype=int) - 1
        for handle in range(self.env.get_num_agents()):
            agent = self.env.agents[handle]
            if agent.state in [TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION]:
                if agent.position is not None:
                    self.agent_positions[agent.position] = handle

    def _shortest_distance_mapper(self):
        if self.shortest_distance_walker is None:
            self.shortest_distance_walker = DeadlockAvoidanceShortestDistanceWalker(self.env)
        self.shortest_distance_walker.clear(self.agent_positions)
        for handle in range(self.env.get_num_agents()):
            agent = self.env.agents[handle]
            if agent.state <= TrainState.MALFUNCTION:
                self.shortest_distance_walker.walk_to_target(handle)

    def _extract_agent_can_move(self):
        self.agent_can_move = {}
        shortest_distance_agent_map, full_shortest_distance_agent_map = self.shortest_distance_walker.getData()
        for handle in range(self.env.get_num_agents()):
            agent = self.env.agents[handle]
            if agent.state < TrainState.DONE:
                if self._check_agent_can_move(handle,
                                              shortest_distance_agent_map[handle],
                                              self.shortest_distance_walker.same_agent_map.get(handle, []),
                                              self.shortest_distance_walker.opp_agent_map.get(handle, []),
                                              full_shortest_distance_agent_map):
                    next_position, next_direction, action, _ = self.shortest_distance_walker.walk_one_step(handle)
                    self.agent_can_move.update({handle: [next_position[0], next_position[1], next_direction, action]})

        if self.show_debug_plot:
            a = np.floor(np.sqrt(self.env.get_num_agents()))
            b = np.ceil(self.env.get_num_agents() / a)
            for handle in range(self.env.get_num_agents()):
                plt.subplot(a, b, handle + 1)
                plt.imshow(full_shortest_distance_agent_map[handle] + shortest_distance_agent_map[handle])
            plt.show(block=False)
            plt.pause(0.01)

    def _check_agent_can_move(self,
                              handle,
                              my_shortest_walking_path,
                              same_agents,
                              opp_agents,
                              full_shortest_distance_agent_map):
        agent_positions_map = (self.agent_positions > -1).astype(int)
        len_opp_agents = len(opp_agents)
        for opp_a in opp_agents:
            opp = full_shortest_distance_agent_map[opp_a]
            delta = ((my_shortest_walking_path - opp - agent_positions_map) > 0).astype(int)
            sum_delta = np.sum(delta)
            if sum_delta < (self.min_free_cell + len_opp_agents):
                return False
        return True

    def save(self, filename):
        pass

    def load(self, filename):
        pass
