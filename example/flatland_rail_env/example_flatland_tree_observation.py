from enum import Enum
from functools import lru_cache
from typing import Optional, List, Union

import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.step_utils.states import TrainState
from flatland_railway_extension.FlatlandGraphBuilder import FlatlandGraphBuilder
from flatland_railway_extension.RailroadSwitchAnalyser import RailroadSwitchAnalyser
from networkx import bfs_tree, dfs_tree

from environment.environment import Environment
from environment.flatland.rail_env import RailEnvironment
from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.deadlock_avoidance_policy import \
    DeadLockAvoidancePolicy
from policy.policy import Policy
from rendering.flatland.flatland_simple_renderer import FlatlandSimpleRenderer
from solver.flatland.flatland_solver import FlatlandSolver

# activate LRU caching
_flatland_tree_observation_lru_cache_functions = []


def _enable_flatland_tree_observation_lru_cache(*args, **kwargs):
    def decorator(func):
        func = lru_cache(*args, **kwargs)(func)
        _flatland_tree_observation_lru_cache_functions.append(func)
        return func

    return decorator


def _send_flatland_tree_observation_data_change_signal_to_reset_lru_cache():
    for func in _flatland_tree_observation_lru_cache_functions:
        func.cache_clear()


class TreeObservationSearchStrategy(Enum):
    DepthFirstSearch = 0
    BreadthFirstSearch = 1


class FlatlandTreeObservation(ObservationBuilder):
    def __init__(self,
                 search_strategy: TreeObservationSearchStrategy = TreeObservationSearchStrategy.BreadthFirstSearch,
                 depth_limit: int = 10):
        super(FlatlandTreeObservation, self).__init__()
        self.search_strategy: TreeObservationSearchStrategy = search_strategy
        self.depth_limit = depth_limit
        self.graph: Union[FlatlandGraphBuilder, None] = None
        self.switchAnalyser: Union[RailroadSwitchAnalyser, None] = None
        self.agents_grid_map: Union[np.ndarray, None] = None
        self.agent_directions = []
        self.agent_state = []
        self._distance_map: Union[np.ndarray, None] = None

    def reset(self):
        self.switchAnalyser = RailroadSwitchAnalyser(self.env)
        self.graph = FlatlandGraphBuilder(self.switchAnalyser, activate_simplified=True)
        self._distance_map = self.env.distance_map.get()
        _send_flatland_tree_observation_data_change_signal_to_reset_lru_cache()

    @staticmethod
    def get_agent_position_and_direction(agent: EnvAgent):
        position = agent.position if agent.position is not None else agent.initial_position
        direction = agent.direction if agent.direction is not None else agent.initial_direction
        return position, direction

    def get_many(self, handles: Optional[List[int]] = None):
        self.agents_grid_map = np.zeros((self.env.rail.height, self.env.rail.width), dtype=int) - 1

        for a in self.env.agents:
            agent: EnvAgent = a
            position, direction = FlatlandTreeObservation.get_agent_position_and_direction(agent)
            if agent.state.is_on_map_state():
                self.agents_grid_map[position] = agent.handle

        return super(FlatlandTreeObservation, self).get_many(handles)

    def get(self, handle: int = 0):
        '''
        The get method returns a feature vector (agent_attr) for each agent and a tree containing the collected features
        at each node. The collected features are the number of other agents per direction, the estimated mean delta
        distance on the "edge" (between decision points), and a flag for whether the branch contains the target cell.
        '''
        agent: EnvAgent = self.env.agents[handle]
        position, direction = FlatlandTreeObservation.get_agent_position_and_direction(agent)
        agent_attr = np.array([handle, agent.state])

        self.env.dev_obs_dict[handle] = set([])

        if not (agent.state.is_on_map_state() or agent.state == TrainState.READY_TO_DEPART):
            return [handle, {'agent_attr': agent_attr,
                             'features': np.array([]),
                             'adjacency': np.array([])}]

        # do calculation only for active agent
        cur_dist = self._distance_map[handle][position][direction]
        node = self._get_mapped_vertex(position, direction)
        search_tree = self._get_search_tree(node[0])

        nodes: List[str] = []
        nodes_idx = {}
        for n in search_tree.nodes:
            if n not in nodes:
                nodes.append(n)
                nodes_idx.update({n: len(nodes) - 1})
        adj = np.zeros((len(search_tree.edges), 2), dtype=int)
        feature = np.zeros((len(nodes), 4 + 1 + 1 + 1))
        visited = []
        for i_edge, edge in enumerate(search_tree.edges):
            edge_node_idx_1 = nodes_idx.get(edge[0])
            edge_node_idx_2 = nodes_idx.get(edge[1])
            adj[i_edge] = [edge_node_idx_1, edge_node_idx_2]

            res = self._get_edge_resource(edge)

            # filter res
            if position in res:
                res = res[res.index(position):]

            # feature extraction
            other_agents = self._find_other_agents(handle, self.agents_grid_map, res)
            for opp_agent in other_agents:
                node = self._get_mapped_vertex(opp_agent.position, opp_agent.direction)
                _, node_dir = self._get_node_position_direction(node[0])
                feature[edge_node_idx_1][int(node_dir)] += 1
            feature[edge_node_idx_1][4] += len(other_agents)
            node_pos_e1, node_dir_e1 = self._get_node_position_direction(edge[0])
            node_pos_e2, node_dir_e2 = self._get_node_position_direction(edge[1])
            dist = min(self._distance_map[handle][node_pos_e1][node_dir_e1],
                       self._distance_map[handle][node_pos_e2][node_dir_e2])

            feature[edge_node_idx_1][5] = dist - cur_dist
            feature[edge_node_idx_1][6] = int(agent.target in res)
            visited = visited + res

        self.env.dev_obs_dict[handle] = set(visited)

        # print(handle, 'nodes', nodes)
        # print(handle, 'adj', adj)
        # print(handle, 'feature', feature)

        return [handle, {'agent_attr': agent_attr,
                         'features': feature,
                         'adjacency': adj}]

    def _find_other_agents(self, handle, agents_grid_map, res) -> List[EnvAgent]:
        ret: List[EnvAgent] = []
        for r in res:
            agent_handle = agents_grid_map[r]
            if agent_handle == -1:
                continue
            if agent_handle == handle:
                continue
            agent = self.env.agents[agent_handle]
            ret.append(agent)
        return ret

    @_enable_flatland_tree_observation_lru_cache(maxsize=1_000_000)
    def _get_node_position_direction(self, node: str):
        node_pos, node_dir = self.graph.get_node_pos_dir(node)
        return node_pos, node_dir

    @_enable_flatland_tree_observation_lru_cache(maxsize=1_000_000)
    def _get_mapped_vertex(self, position, direction):
        node = self.graph.get_mapped_vertex(position, direction)
        return node

    @_enable_flatland_tree_observation_lru_cache(maxsize=1_000_000)
    def _get_edge_resource(self, edge):
        res = self.graph.get_edge_resource(edge)
        return res

    @_enable_flatland_tree_observation_lru_cache(maxsize=1_000_000)
    def _get_search_tree(self, node):
        if self.search_strategy == TreeObservationSearchStrategy.BreadthFirstSearch:
            search_tree = bfs_tree(self.graph.get_graph(), node, depth_limit=self.depth_limit)
        else:
            search_tree = dfs_tree(self.graph.get_graph(), node, depth_limit=self.depth_limit)

        return search_tree


def create_deadlock_avoidance_policy(environment: Environment, action_space: int, show_debug_plot=False) -> Policy:
    return DeadLockAvoidancePolicy(environment.get_raw_env(), action_space, enable_eps=False,
                                   show_debug_plot=show_debug_plot)


if __name__ == "__main__":
    do_rendering = True

    env = RailEnvironment(
        obs_builder_object=FlatlandTreeObservation(
            search_strategy=TreeObservationSearchStrategy.BreadthFirstSearch,
            depth_limit=10
        ),
        number_of_agents=5)

    solver_deadlock = FlatlandSolver(env,
                                     create_deadlock_avoidance_policy(env, env.get_action_space()),
                                     FlatlandSimpleRenderer(env) if do_rendering else None)
    solver_deadlock.perform_training(max_episodes=2)
