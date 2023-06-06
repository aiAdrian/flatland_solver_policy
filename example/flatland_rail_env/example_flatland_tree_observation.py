from enum import Enum
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
        self.agent_positions = []
        self.agent_directions = []
        self.agent_state = []

    def reset(self):
        print("OptimisedTreeObs.reset")
        print("- RailroadSwitchAnalyser", end=" ")
        self.switchAnalyser = RailroadSwitchAnalyser(self.env)
        print("ok.")
        # TODO FlatlandGraphBuild has new argument: activate_simplified / Install latest version and replace here
        print("- FlatlandGraphBuilder", end=" ")
        self.graph = FlatlandGraphBuilder(self.switchAnalyser)
        self.graph.activate_simplified()
        print("ok.")

    @staticmethod
    def get_agent_position_and_direction(agent: EnvAgent):
        position = agent.position if agent.position is not None else agent.initial_position
        direction = agent.direction if agent.direction is not None else agent.initial_direction
        return position, direction

    def get_many(self, handles: Optional[List[int]] = None):
        self.agent_positions = np.zeros(self.env.get_num_agents(), dtype=[('x', int), ('y', int)])
        self.agent_directions = np.zeros(self.env.get_num_agents())
        self.agent_state = np.zeros(self.env.get_num_agents())

        for a in self.env.agents:
            agent: EnvAgent = a
            position, direction = FlatlandTreeObservation.get_agent_position_and_direction(agent)
            if agent.state.is_on_map_state():
                self.agent_positions[agent.handle] = position
                self.agent_directions[agent.handle] = direction
                self.agent_state[agent.handle] = agent.state
            else:
                self.agent_positions[agent.handle] = (-1, -1)

        return super(FlatlandTreeObservation, self).get_many(handles)

    def get(self, handle: int = 0):
        agent: EnvAgent = self.env.agents[handle]
        position, direction = FlatlandTreeObservation.get_agent_position_and_direction(agent)
        cur_dist = self.env.distance_map.get()[handle][position][direction]
        agent_attr = np.array([handle, agent.state])

        self.env.dev_obs_dict[handle] = set([])

        if not (agent.state.is_on_map_state() or agent.state == TrainState.READY_TO_DEPART):
            return [handle, {'agent_attr': agent_attr,
                             'forest': np.array([]),
                             'adjacency': np.array([])}]

        # do calculation only for active agent
        node = self.graph.get_mapped_vertex(position, direction)
        if self.search_strategy == TreeObservationSearchStrategy.BreadthFirstSearch:
            search_tree = bfs_tree(self.graph.get_graph(), node[0], depth_limit=self.depth_limit)
        else:
            search_tree = dfs_tree(self.graph.get_graph(), node[0], depth_limit=self.depth_limit)

        nodes: List[str] = []
        for n in search_tree.nodes:
            if n not in nodes:
                nodes.append(n)
        adj = []
        feature = np.zeros((len(nodes), 4 + 1 + 1 + 1))
        for e in search_tree.edges:
            e1 = nodes.index(e[0])
            e2 = nodes.index(e[1])
            res = self.graph.get_edge_resource(e)
            x_res = np.array(res, dtype=[('x', int), ('y', int)])
            other_agents = np.in1d(self.agent_positions, x_res)
            other_agents[handle] = False
            for opp_agent_dir in self.agent_directions[other_agents]:
                feature[e1][int(opp_agent_dir)] += 1
            feature[e1][4] += len(self.agent_positions[other_agents])
            mean_dist = 0

            # filter res
            if position in res:
                res = res[res.index(position):]

            for r in res:
                mean_dist = np.min(self.env.distance_map.get()[handle][r])
            feature[e1][5] = (mean_dist / (len(res) if len(res) > 0 else 1)) - cur_dist
            feature[e1][6] = int(agent.target in res)
            adj.append((e1, e2))
            self.env.dev_obs_dict[handle] = self.env.dev_obs_dict[handle].union(set(res))

        # print(handle, 'nodes', nodes)
        # print(handle, 'adj', adj)
        # print(handle, 'feature', feature)

        return [handle, {'agent_attr': agent_attr,
                         'forest': feature,
                         'adjacency': np.array(adj)}]


def create_deadlock_avoidance_policy(environment: Environment, action_space: int, show_debug_plot=False) -> Policy:
    return DeadLockAvoidancePolicy(environment.get_raw_env(), action_space, enable_eps=False,
                                   show_debug_plot=show_debug_plot)


if __name__ == "__main__":
    env = RailEnvironment(
        obs_builder_object=FlatlandTreeObservation(
            search_strategy=TreeObservationSearchStrategy.BreadthFirstSearch,
            depth_limit=10
        ),
        number_of_agents=20)

    solver_deadlock = FlatlandSolver(env,
                                     create_deadlock_avoidance_policy(env, env.get_action_space()),
                                     FlatlandSimpleRenderer(env))
    solver_deadlock.perform_training(max_episodes=2)
