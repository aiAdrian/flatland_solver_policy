from collections import namedtuple
from enum import Enum
from functools import lru_cache
from typing import Optional, List, Union, Tuple

import networkx
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.fast_methods import fast_count_nonzero, fast_argmax
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState
from flatland_railway_extension.FlatlandGraphBuilder import FlatlandGraphBuilder
from flatland_railway_extension.RailroadSwitchAnalyser import RailroadSwitchAnalyser
from matplotlib import pyplot as plt
from networkx import bfs_tree, dfs_tree

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


class TreeObservationReturnType(Enum):
    Flatten = 0
    Tree = 1


TreeObservationData = namedtuple('TreeObservationData',
                                 ['handle', 'agent',
                                  'agent_attr', 'features', 'adjacency', 'actions',
                                  'nodes', 'edges', 'search_tree',
                                  'nodes_type'])

TREE_EDGE_FEATURE_SIZE = 10


class FlatlandTreeObservation(ObservationBuilder):
    def __init__(self,
                 search_strategy: TreeObservationSearchStrategy = TreeObservationSearchStrategy.BreadthFirstSearch,
                 observation_return_type: TreeObservationReturnType = TreeObservationReturnType.Tree,
                 depth_limit: int = 10,
                 observation_depth_limit: int = 10,
                 observation_depth_limit_discount: float = 0.75,
                 activate_simplified=True,
                 render_debug_tree=False):
        super(FlatlandTreeObservation, self).__init__()
        self.search_strategy: TreeObservationSearchStrategy = search_strategy
        self.observation_return_type = observation_return_type
        self.depth_limit = max(1, depth_limit)
        self.observation_depth_limit = min(self.depth_limit, observation_depth_limit)
        self.observation_depth_limit_discount = observation_depth_limit_discount
        self.tree_feature_size = TREE_EDGE_FEATURE_SIZE
        self.render_debug_tree = render_debug_tree

        self.switchAnalyser: Union[RailroadSwitchAnalyser, None] = None
        self.graph: Union[FlatlandGraphBuilder, None] = None
        self.agents_grid_map: Union[np.ndarray, None] = None

        self.agent_directions = []
        self.agent_state = []
        self._distance_map: Union[np.ndarray, None] = None
        self._activate_simplified = activate_simplified

    def reset(self):
        self.switchAnalyser = RailroadSwitchAnalyser(self.env)
        self.graph = FlatlandGraphBuilder(self.switchAnalyser,
                                          activate_simplified=self._activate_simplified,
                                          keep_switch_neighbors_at_simplification=True
                                          )
        self._distance_map = self.env.distance_map.get()
        _send_flatland_tree_observation_data_change_signal_to_reset_lru_cache()

    def render_search_tree(self, obs: TreeObservationData):
        plt.figure()

        points = []
        for v in range(self.env.rail.width):
            for u in range(self.env.rail.height):
                if self.env.rail.grid[(u, v)] > 0:
                    points.append([u, v])
        x, y = np.array(points).T
        plt.scatter(1000 * np.array(y), -1000 * np.array(x), color='red')

        points = []
        position, direction = FlatlandTreeObservation.get_agent_position_and_direction(obs.agent)
        for i_edge, edge in enumerate(obs.search_tree.edges):
            res = self._get_edge_resource(edge)
            if position in res:
                res = res[res.index(position):]
            for r in res:
                points.append([r[0], r[1]])
        x, y = np.array(points).T
        plt.scatter(1000 * np.array(y), -1000 * np.array(x), color='blue')

        options = {
            "font_size": 4,
            "node_size": 150,
            "edgecolors": "black",
            "linewidths": 2,
            "width": 2,
        }

        pos = {}
        nodes_color = []
        for idx, n in enumerate(obs.nodes):
            p, _ = self._get_node_position_direction(n)
            pos.update({n: [1000 * p[1], -1000 * p[0]]})
            binary = obs.nodes_type[idx]
            color_value = float(sum(val * (2 ** idx) for idx, val in enumerate(reversed(binary))))
            nodes_color.append(color_value)

        networkx.draw(obs.search_tree,
                      pos=pos,
                      with_labels=True,
                      node_color=nodes_color,
                      cmap=plt.cm.Blues,
                      alpha=0.8,
                      **options
                      )

        edge_labels = dict([((edge[0], edge[1]),
                             f'{self._get_edge_action(edge)}'.replace('[<RailEnvActions.', '').replace('>]', ''))
                            for edge in obs.search_tree.edges])

        networkx.draw_networkx_edge_labels(
            obs.search_tree,
            pos=pos,
            edge_labels=edge_labels,
            label_pos=0.5,
            font_color='red',
            font_size=6,
        )

        plt.scatter(1000 * np.array([position[1]]), -1000 * np.array([position[0]]), color='orange')

        plt.axis('off')
        plt.show()

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
        agent_attr = self._get_shortest_walk_action_one_hot_encoded(agent.handle, position, direction)
        agent_attr.append(agent.state)

        self.env.dev_obs_dict[handle] = set([])

        if not (agent.state.is_on_map_state() or agent.state == TrainState.READY_TO_DEPART):
            return self._transform_observation(
                TreeObservationData(
                    handle=handle,
                    agent=agent,
                    agent_attr=agent_attr,
                    features=np.array([]),
                    adjacency=np.array([]),
                    actions=np.array([]),
                    nodes=[],
                    edges=[],
                    search_tree=None,
                    nodes_type=[]
                ))

        # do calculation only for active agent
        cur_dist = self._distance_map[handle][position][direction]
        node = self._get_mapped_vertex(position, direction)
        search_tree = self._get_search_tree(node[0])

        nodes: List[str] = []
        nodes_type = []
        nodes_idx = {}
        for n in search_tree.nodes:
            if n not in nodes:
                nodes.append(n)
                nodes_idx.update({n: len(nodes) - 1})
                n_pos, n_dir = self._get_node_position_direction(n)
                is_dead_end = self.graph.railroad_switch_analyser.is_dead_end(n_pos)
                is_diamond_crossing = self.graph.railroad_switch_analyser.is_diamond_crossing(n_pos)
                agent_at_railroad_switch, \
                    agent_near_to_railroad_switch, \
                    agent_at_railroad_switch_cell, \
                    agent_near_to_railroad_switch_cell = self._check_agent_decision(n_pos, n_dir)
                nt = self._create_node_types(is_dead_end,
                                             is_diamond_crossing,
                                             agent_at_railroad_switch,
                                             agent_near_to_railroad_switch,
                                             agent_at_railroad_switch_cell,
                                             agent_near_to_railroad_switch_cell)
                nodes_type.append(nt)

        adj = np.zeros((len(search_tree.edges), 2), dtype=int)
        actions = np.zeros(len(search_tree.edges))
        edges = []
        feature = np.zeros((len(nodes), self.tree_feature_size))
        visited = []
        for i_edge, edge in enumerate(search_tree.edges):
            edges.append(edge)
            edge_node_idx_1 = nodes_idx.get(edge[0])
            edge_node_idx_2 = nodes_idx.get(edge[1])
            adj[i_edge] = [edge_node_idx_1, edge_node_idx_2]
            actions[i_edge] = self._get_edge_action(edge)[0]

            res = self._get_edge_resource(edge)

            # filter res
            if position in res:
                res = res[res.index(position):]

            # ---------------------------------------------------------------------------
            # feature extraction
            # idx
            #  0 : nbr of other agents in direction 0
            #  1 : nbr of other agents in direction 1
            #  2 : nbr of other agents in direction 2
            #  3 : nbr of other agents in direction 3
            #  4 : nbr of other agents
            #  5 : delta distance (min(edge_distance) - current_distance)
            #  6 : edge contains agent's target
            #  7 : action to enter the edge (MOVE_LEFT)
            #  8 : action to enter the edge (MOVE_FORWARD)
            #  9 : action to enter the edge (MOVE_RIGHT)
            # ---------------------------------------------------------------------------
            other_agents = self._find_other_agents(handle, self.agents_grid_map, res)
            cnt_other_agents = 0
            for opp_agent in other_agents:
                if opp_agent.position in res:
                    node = self._get_mapped_vertex(opp_agent.position, opp_agent.direction)
                    _, node_dir = self._get_node_position_direction(node[0])
                    transformed_node_dir = (node_dir + direction + 2) % 4
                    feature[edge_node_idx_1][int(transformed_node_dir)] += 1
                    cnt_other_agents += 1
            feature[edge_node_idx_1][4] += cnt_other_agents
            node_pos_e1, node_dir_e1 = self._get_node_position_direction(edge[0])
            node_pos_e2, node_dir_e2 = self._get_node_position_direction(edge[1])
            dist = min(self._distance_map[handle][node_pos_e1][node_dir_e1],
                       self._distance_map[handle][node_pos_e2][node_dir_e2])

            feature[edge_node_idx_1][5] = (1.0 - int(agent.target in res)) * (dist - cur_dist)
            feature[edge_node_idx_1][6] = int(agent.target in res)
            feature[edge_node_idx_1][7] = int(actions[i_edge] == RailEnvActions.MOVE_LEFT)
            feature[edge_node_idx_1][8] = int(actions[i_edge] == RailEnvActions.MOVE_FORWARD)
            feature[edge_node_idx_1][9] = int(actions[i_edge] == RailEnvActions.MOVE_RIGHT)
            # ---------------------------------------------------------------------------

            visited = visited + res

        self.env.dev_obs_dict[handle] = set(visited)

        # print(handle, 'nodes', nodes)
        # print(handle, 'adj', adj)
        # print(handle, 'feature', feature)

        return self._transform_observation(
            TreeObservationData(
                handle=handle,
                agent=agent,
                agent_attr=agent_attr,
                features=feature,
                adjacency=adj,
                actions=actions,
                nodes=nodes,
                edges=edges,
                search_tree=search_tree,
                nodes_type=nodes_type
            ))

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

    def _get_transition_action(self, position, direction, next_direction, next_position):
        possible_transitions = self.env.rail.get_transitions(*position, direction)
        num_transitions = fast_count_nonzero(possible_transitions)
        if num_transitions == 1:
            return RailEnvActions.MOVE_FORWARD
        else:
            a = [0] * 4
            for new_direction in [(direction + i) % 4 for i in range(-1, 2)]:
                if possible_transitions[new_direction]:
                    new_position = get_new_position(position, new_direction)
                    if next_position == new_position and next_direction == new_direction:
                        a[new_direction] = 1
            return fast_argmax(a)

    def _get_shortest_walk_action_one_hot_encoded(self, handle, position, direction):

        ret = [0, 0, 0]
        possible_transitions = self.env.rail.get_transitions(*position, direction)
        num_transitions = fast_count_nonzero(possible_transitions)
        if num_transitions == 1:
            ret[RailEnvActions.MOVE_FORWARD] = 1
            return ret

        distance_map = self.env.distance_map.get()
        min_distances = []
        for new_direction in [(direction + i) % 4 for i in range(-1, 2)]:
            if possible_transitions[new_direction]:
                new_position = get_new_position(position, new_direction)
                min_distances.append(
                    distance_map[handle, new_position[0], new_position[1], new_direction]
                )
            else:
                min_distances.append(np.inf)

        ret = [1 if x == min(min_distances) else 0 for x in min_distances]
        return ret

    def _transform_observation(self, obs: TreeObservationData):
        if self.observation_return_type == TreeObservationReturnType.Tree:
            return obs

        flatten_obj = self.search_tree_flatten(obs)

        if self.render_debug_tree:
            if obs.search_tree is not None and obs.agent.state.is_on_map_state():
                if len(obs.adjacency) > self.depth_limit:
                    self.render_search_tree(obs)

        return np.concatenate([obs.agent_attr, flatten_obj])

    def search_tree_flatten(self, obs: TreeObservationData):
        children_pre_nodes = 3
        flatten_obs = np.zeros((children_pre_nodes ** (self.observation_depth_limit + 1)) * self.tree_feature_size)

        if obs.search_tree is not None:
            parents = obs.adjacency[:, 0]
            children = obs.adjacency[:, 1]

            # get node depth
            shortest_path_to_root = networkx.shortest_path_length(obs.search_tree, obs.nodes[parents[0]])
            node_depths = [shortest_path_to_root.get(n) for i, n in enumerate(obs.nodes)]

            # compute for each node (child, the idx in the flatten array)
            node_idx = {}
            node_idx.update({parents[0]: 0})
            for i_p, parent in enumerate(parents):
                child = children[i_p]
                parent_node_id = node_idx.get(parent)
                idx_offset = 0
                if i_p > 0:
                    edge = (obs.nodes[parent], obs.nodes[child])
                    idx_offset = max(0, min(children_pre_nodes, int(self._get_edge_action(edge)[0]) - 1))
                calculated_idx = (children_pre_nodes * (parent_node_id + 1) - 1) + idx_offset
                node_idx.update({child: calculated_idx})

            # reduce the tree
            for n_i, n in enumerate(obs.nodes):
                if node_depths[n_i] == self.observation_depth_limit:
                    sub_tree = networkx.dfs_tree(obs.search_tree, n)
                    idx_root = obs.nodes.index(n)
                    for i, nd in enumerate(sub_tree.nodes):
                        if i > 0:
                            idx = obs.nodes.index(nd)
                            obs.features[idx_root] += obs.features[idx] * np.exp(
                                self.observation_depth_limit
                                - self.observation_depth_limit_discount * node_depths[idx]
                            )

            # burn the data into the flatten structure
            for n_idx in range(len(obs.nodes)):
                level = node_depths[n_idx]
                # only use for flatten nodes with level below equal observation_depth_limit
                if level <= self.observation_depth_limit:
                    x = self.tree_feature_size * node_idx.get(n_idx)
                    flatten_obs[x:(x + self.tree_feature_size)] = obs.features[n_idx]

        return flatten_obs

    @_enable_flatland_tree_observation_lru_cache(maxsize=1_000_000)
    def _check_agent_decision(self, n_pos, n_dir):
        agent_at_railroad_switch, agent_near_to_railroad_switch, \
            agent_at_railroad_switch_cell, agent_near_to_railroad_switch_cell = \
            self.graph.railroad_switch_analyser.check_agent_decision(n_pos, n_dir)
        return agent_at_railroad_switch, agent_near_to_railroad_switch, \
            agent_at_railroad_switch_cell, agent_near_to_railroad_switch_cell

    @_enable_flatland_tree_observation_lru_cache(maxsize=1_000_000)
    def extract_node_types(self, node_type) -> (int, int, int, int, int, int):
        is_dead_end = node_type[0]
        is_diamond_crossing = node_type[1]
        agent_at_railroad_switch = node_type[2]
        agent_near_to_railroad_switch = node_type[3]
        agent_at_railroad_switch_cell = node_type[4]
        agent_near_to_railroad_switch_cell = node_type[5]
        return is_dead_end, is_diamond_crossing, \
            agent_at_railroad_switch, agent_near_to_railroad_switch, \
            agent_at_railroad_switch_cell, agent_near_to_railroad_switch_cell

    @_enable_flatland_tree_observation_lru_cache(maxsize=1_000_000)
    def _create_node_types(self, is_dead_end: bool, is_diamond_crossing: bool,
                           agent_at_railroad_switch: bool, agent_near_to_railroad_switch: bool,
                           agent_at_railroad_switch_cell: bool, agent_near_to_railroad_switch_cell: bool):
        return [int(is_dead_end), int(is_diamond_crossing),
                int(agent_at_railroad_switch), int(agent_near_to_railroad_switch),
                int(agent_at_railroad_switch_cell), int(agent_near_to_railroad_switch_cell)]

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
    def _get_edge_action(self, edge) -> List[Tuple[int, int]]:
        edge_data = self.graph.get_graph().get_edge_data(edge[0], edge[1])
        return edge_data.get('action')

    @_enable_flatland_tree_observation_lru_cache(maxsize=1_000_000)
    def _get_search_tree(self, node):
        if self.search_strategy == TreeObservationSearchStrategy.BreadthFirstSearch:
            search_tree = bfs_tree(self.graph.get_graph(), node, depth_limit=self.depth_limit)
        else:
            search_tree = dfs_tree(self.graph.get_graph(), node, depth_limit=self.depth_limit)

        return search_tree
