from typing import Callable, Type, List, Union

import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from typing import Optional, List, Any, Tuple, Set
from flatland.envs.rail_env import RailEnv, RailEnvActions
import heapq  # Für die A*-Prioritätswarteschlange
from flatland.core.grid.grid4_utils import get_new_position

from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState
from flatland_railway_extension.RailroadSwitchAnalyser import RailroadSwitchAnalyser
from flatland.envs.fast_methods import fast_count_nonzero, fast_argmax
from flatland.envs.step_utils.states import TrainState

from environment.environment import Environment
from example.flatland_rail_env.flatland_rail_env_persister import RailEnvironmentPersistable
from observation.flatland.flatland_tree_observation.flatland_tree_observation import FlatlandTreeObservation, \
    TreeObservationSearchStrategy, TreeObservationReturnType
from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.deadlock_avoidance_policy import \
    DeadLockAvoidancePolicy
from policy.learning_policy.learning_policy import LearningPolicy
from rendering.flatland.flatland_simple_renderer import FlatlandSimpleRenderer
from solver.flatland.flatland_solver import FlatlandSolver
from solver.multi_agent_base_solver import RewardList, TerminalList, InfoDict
from policy.policy import Policy
from policy.learning_policy.ppo_policy.ppo_agent import PPOPolicy, PPO_Param
from utils.training_evaluation_pipeline import policy_creator_list
from utils.training_evaluation_pipeline import create_td3_policy, create_a2c_policy, create_ppo_policy, create_dddqn_policy, create_random_policy
from utils.training_evaluation_pipeline import policy_creator_list
from utils.training_evaluation_pipeline import create_td3_policy, create_a2c_policy, create_ppo_policy, create_dddqn_policy, create_random_policy



# Enforce disable GPU
import torch
torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import heapq  # Für die A*-Prioritätswarteschlange

class ExperimentalAStarObs(ObservationBuilder):
    """
    Sehr reiche Observation nur durch Lesenzugriff auf Environment.
    Leistungsfokus: fast lokale Checks + begrenzte A*-Tiefe.
    Liefert:
      - globale/relative Zielinfos
      - lokales Motion-HUD (Transitions, besetzt, cell-type)
      - Lookahead-Pfad (max_path_length) mit pro-zelle Features
      - Agenten-Interaktions-Features (distanz, voraussichtliche Kollision)
      - einfache Normalisierung/Clipping für RL-Netze
    Nur Observation wird verändert, keine Policy/Reward.
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

        self._vec_len = 15
        self.observation_space = np.zeros(self._vec_len, dtype=np.float32)

    def reset(self):
        pass

    @staticmethod
    def get_pos_dir(agent: EnvAgent):
        pos = agent.position if agent.position is not None else agent.initial_position
        dir = agent.direction if agent.direction is not None else agent.initial_direction
        return pos, dir

    def get_many(self, handles: List[int]):
        # build agent map fast for occupancy lookups
        h, w = self.env.height, self.env.width
        self.agent_map = np.full((h, w), -1, dtype=int)
        for agent in self.env.agents:
            pos, _ = self.get_pos_dir(agent)
            if pos is None:
                continue
            if agent.state.is_on_map_state():
                self.agent_map[pos] = agent.handle
        return super().get_many(handles)

    def get(self, handle: int) -> np.ndarray:
        distance_map = self.env.distance_map.get()

        agent = self.env.agents[handle]
        pos, dir = self.get_pos_dir(agent)
        target = agent.target

        if pos is None or target is None:
            return np.zeros(self._vec_len, dtype=np.float32)


        vec = []

        # 1) Distanz (clipped + normalized)
        dist = distance_map[handle, pos[0], pos[1], dir]
        v = distance_map[handle]
        max_dist = np.max(v[v!=np.inf])
        dist_clipped = min(dist, max_dist)
        vec.append(dist_clipped / float(max_dist))  # 0..1

        dist = distance_map[handle, pos[0], pos[1], dir]
        current_dist_value = 1.0 / (1.0 + np.exp(-dist) if dist != np.inf else -1 )
        vec.append(current_dist_value)

        # 2) Zielrichtung -> switch one-step ahead
        transitions = self.env.rail.get_transitions(*pos, dir)
        for nd in [(dir + i) % 4 for i in range(-1, 2)]:  # left, straight, right
            if transitions[nd]:
                npos = get_new_position(pos, nd)
                dist = distance_map[handle, npos[0], npos[1], nd]
                dist_clipped = min(dist, max_dist)
                vec.append(dist_clipped / float(max_dist))  # 0..1

                # A* with cost limit and respecting relative moves (left, straight, right)
                path, visited_info = self._a_star_limited(handle, pos, target, dir, self.max_path_length, self.lookahead_cost_limit)
                # 4) Anzahl Agenten auf Pfad (clipped & normalized)
                agents_on_path = self._count_agents(path, handle)
                vec.append(min(agents_on_path, len(self.env.agents)) / len(self.env.agents))

                # 8) distance to next agent in forward direction (clipped & normalized)
                vec.append(self._distance_to_next_agent(pos, dir) / float(self.max_agent_dist))
            else:
                vec.append(-1)
                vec.append(-1)
                vec.append(-1)

        # 5) Agentenstatus bits
        vec.append(1.0 if agent.state == TrainState.MOVING else 0.0)
        vec.append(1.0 if agent.state == TrainState.WAITING else 0.0)
        vec.append(1.0 if agent.state == TrainState.DONE or agent.state == TrainState.WAITING else 0.0)
        vec.append(1.0 if pos == target else 0.0)

        # return fixed-size numpy vector
        arr = np.array(vec, dtype=np.float32)
        if arr.shape[0] != self._vec_len:
            # safety padding or trimming
            if arr.shape[0] < self._vec_len:
                pad = np.zeros(self._vec_len - arr.shape[0], dtype=np.float32)
                arr = np.concatenate([arr, pad])
            else:
                arr = arr[:self._vec_len]

        return arr

    # ---------- helper routines ----------
    def _a_star_limited(self, handle, start: Tuple[int, int], goal: Tuple[int, int],
                        start_dir: int, max_nodes: int, cost_limit: int):

        distance_map = self.env.distance_map.get()
        start_dist = distance_map[handle, start[0],start[1], start_dir]
        if start_dist is None:
            start_dist = -1

        # A* that explores left/straight/right only, returns positions up to goal or truncated path
        pq = [(0 + self._heuristic(start, goal), 0, start, start_dir)]
        visited: Dict[Tuple[int, int, int], int] = {}
        parents: Dict[Tuple[int, int, int], Tuple[Tuple[int, int], int]] = {}
        nodes_popped = 0
        info = []

        while pq:
            _, cost, pos, dir = heapq.heappop(pq)
            nodes_popped += 1
            if nodes_popped > cost_limit:
                break
            if pos == goal:
                # reconstruct
                path = []
                cur = (pos, dir)
                while cur[0] != start:
                    path.append(cur[0])
                    cur = parents.get((cur[0], cur[1]), (start, start_dir))
                path.reverse()
                return path[:max_nodes], info
            if (pos, dir) in visited and cost >= visited[(pos, dir)]:
                continue
            visited[(pos, dir)] = cost
            transitions = self.env.rail.get_transitions(*pos, dir)
            for nd in [(dir + i) % 4 for i in range(-1, 2)]:  # left, straight, right
                if transitions[nd]:
                    npos = get_new_position(pos, nd)

                    dist = distance_map[handle, npos[0], npos[1], nd]
                    if dist is None:
                        dist = -1

                    ah = self.agent_map[npos];
                    has_other_agent = ah != handle and ah != -1
                    has_opp_agent = -1
                    if has_other_agent:
                        a = self.env.agents[ah]
                        opp_pos, opp_dir = self.get_pos_dir(a)
                        has_opp_agent = 1 if opp_dir != nd else 0
                    info.append(1 if dist - start_dist else 0)
                    info.append(1 if has_other_agent else 0)
                    info.append(has_opp_agent)

                    new_cost = cost + 1
                    heur = self._heuristic(npos, goal)
                    total = new_cost + heur
                    if total > cost_limit:
                        continue
                    if ((npos, nd) not in visited) or new_cost < visited.get((npos, nd), 1e9):
                        parents[(npos, nd)] = (pos, dir)
                        heapq.heappush(pq, (new_cost + heur, new_cost, npos, nd))
                else:
                    info.append(-1)
                    info.append(-1)
                    info.append(-1)
        # fail: return best-effort path by following parent chain from closest visited to start
        # find visited node closest to goal by heuristic
        best = None
        best_h = 1e9
        for (p, d), c in visited.items():
            h = self._heuristic(p, goal)
            if h < best_h:
                best_h = h
                best = (p, d)
        if best is None:
            return [], info
        path = []
        cur = best
        while cur[0] != start and len(path) < max_nodes:
            path.append(cur[0])
            cur = parents.get((cur[0], cur[1]), (start, start_dir))
        path.reverse()
        return path, info

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def _count_agents(self, path: List[Tuple[int, int]], handle: int) -> int:
        if not path:
            return 0
        s = set(path)
        cnt = 0
        for i, a in enumerate(self.env.agents):
            if i == handle:
                continue
            p, _ = self.get_pos_dir(a)
            if p is not None and p in s:
                cnt += 1
        return cnt

    def _distance_to_next_agent(self, pos: Tuple[int, int], dir: int) -> float:
        # walk forward until find agent or hit bounds; clip at max_agent_dist
        next_pos = pos
        for d in range(1, self.max_agent_dist + 1):
            next_pos = get_new_position(next_pos, dir)
            x, y = next_pos
            if not (0 <= x < self.env.height and 0 <= y < self.env.width):
                return float(d)
            if self.agent_map[next_pos] >= 0:
                return float(d)
        return float(self.max_agent_dist)


def create_obs_builder_object():
    return ExperimentalAStarObs()




def create_deadlock_avoidance_policy(environment: Environment,
                                     action_space: int,
                                     show_debug_plot=False) -> DeadLockAvoidancePolicy:
    return DeadLockAvoidancePolicy(environment.get_raw_env(),
                                   action_space,
                                   enable_eps=False,
                                   show_debug_plot=show_debug_plot)

class DecisionPointPPOPolicy(PPOPolicy):
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 in_parameters: Union[PPO_Param, None] = None,
                 use_deadlock_avoidance_policy = False,
                 use_decision_point_heuristic = True):
        self.deadlock_avoidance_policy = None
        self.use_deadlock_avoidance_policy = use_deadlock_avoidance_policy
        self.use_decision_point_heuristic = use_decision_point_heuristic
        super(DecisionPointPPOPolicy, self).__init__(state_size, action_size, in_parameters)
        self._env: Union[Environment, None] = None
        self.switchAnalyser: Union[RailroadSwitchAnalyser, None] = None

    def get_name(self):
        if not self.use_decision_point_heuristic:
            if not self.use_deadlock_avoidance_policy:
                return "PPOPolicy"
            return "PPOPolicy_DLA"
        if self.use_deadlock_avoidance_policy:
            return self.__class__.__name__ + "_DLA"
        return self.__class__.__name__


    def step(self, handle, state, action, reward, next_state, done):
        super(DecisionPointPPOPolicy, self).step(handle, state, action, reward, next_state, done)
        if self.use_deadlock_avoidance_policy:
            if self.deadlock_avoidance_policy is not None:
                self.deadlock_avoidance_policy.step(handle, state, action, reward, next_state, done)

    def start_step(self, train):
        super(DecisionPointPPOPolicy, self).start_step(train)
        if self.use_deadlock_avoidance_policy:
            self.deadlock_avoidance_policy.start_step(train)

    def reset(self, env: Environment):
        self._env = env
        self.switchAnalyser = RailroadSwitchAnalyser(env.raw_env)
        super(DecisionPointPPOPolicy, self).reset(env)
        if self.use_deadlock_avoidance_policy:
            if self.deadlock_avoidance_policy is None:
                self.deadlock_avoidance_policy = create_deadlock_avoidance_policy(env, self.action_size)
            self.deadlock_avoidance_policy.reset(env)


    @staticmethod
    def _get_agent_position_and_direction(agent: EnvAgent):
        position = agent.position if agent.position is not None else agent.initial_position
        direction = agent.direction if agent.direction is not None else agent.initial_direction
        return position, direction

    def act(self, handle: int, state, eps=0.):
        agent: EnvAgent = self._env.raw_env.agents[handle]
        position, direction = self._get_agent_position_and_direction(agent)
        agent_at_railroad_switch, agent_near_to_railroad_switch, \
            agent_at_railroad_switch_cell, agent_near_to_railroad_switch_cell = \
            self.switchAnalyser.check_agent_decision(position=position, direction=direction)

        # only if the agent is moving
        if agent.state == TrainState.MOVING and not self.use_decision_point_heuristic:
            # when the agent is moving and the agent is not at a decision point - best option is just move forward
            # near to all switches are important:
            # (1) fork
            #
            #                |   /  |[---][---][
            #                |  /   |
            #      [---][-A-]| /    |[---][---][--
            #      ->->->->-> switch >->->->->->
            #
            # (2) fusion
            #
            #      -][---][-B-]| \    |
            #                  |  \   |
            #      -][---][-C-]|   \  |[---][---
            #      ->->->->->-> switch >->->->->->
            #
            # A, B, C are cases where the agent should not just walk forward (due deadlock)
            # switch as well
            if not agent_at_railroad_switch and not agent_near_to_railroad_switch_cell:
                return RailEnvActions.MOVE_FORWARD

        action = super(DecisionPointPPOPolicy, self).act(handle, state, eps)
        if self.use_deadlock_avoidance_policy:
            if agent.state.is_on_map_state():
                if action == RailEnvActions.MOVE_FORWARD:
                    dla_action = self.deadlock_avoidance_policy.act(handle, state, eps)
                    return dla_action
                    # if RailEnvActions.STOP_MOVING == dla_action:
                    #     return RailEnvActions.STOP_MOVING

        return action

def create_dp_ppo_policy(observation_space: int, action_space: int) -> LearningPolicy:
    print('>> create_ppo_policy')
    print('   - observation_space:', observation_space)
    print('   - action_space:', action_space)
    ppo_param = PPO_Param(hidden_size=128,
                          buffer_size=16_000,
                          buffer_min_size=0,
                          batch_size=512,
                          learning_rate=0.5e-4,
                          discount=0.75,
                          use_replay_buffer=True,
                          use_gpu=True)
    return DecisionPointPPOPolicy(observation_space, action_space, ppo_param, False)


def create_dp_ppo_policy_dla(observation_space: int, action_space: int) -> LearningPolicy:
    print('>> create_ppo_policy_dla')
    print('   - observation_space:', observation_space)
    print('   - action_space:', action_space)
    ppo_param = PPO_Param(hidden_size=512,
                          buffer_size=8_000,
                          buffer_min_size=0,
                          batch_size=512,
                          learning_rate=0.5e-4,
                          discount=0.75,
                          use_replay_buffer=True,
                          use_gpu=True)
    return DecisionPointPPOPolicy(observation_space, action_space, ppo_param, True)


def create_ppo_policy_dla(observation_space: int, action_space: int) -> LearningPolicy:
    print('>> create_ppo_policy_dla')
    print('   - observation_space:', observation_space)
    print('   - action_space:', action_space)
    ppo_param = PPO_Param(hidden_size=512,
                          buffer_size=8_000,
                          buffer_min_size=0,
                          batch_size=512,
                          learning_rate=0.5e-4,
                          discount=0.75,
                          use_replay_buffer=True,
                          use_gpu=True)
    return DecisionPointPPOPolicy(observation_space, action_space, ppo_param, True, False)


global reward_signal_updated
reward_signal_updated = None


def flatland_reward_shaper(reward: RewardList, terminal: TerminalList, info: InfoDict, env: Environment) -> List[float]:
    global reward_signal_updated
    distance_map = env.raw_env.distance_map.get()
    if env.raw_env._elapsed_steps < 5:
        reward_signal_updated = None
    if reward_signal_updated is None:
        reward_signal_updated = np.zeros(len(env.raw_env.agents))
    for i, agent in enumerate(env.raw_env.agents):
        reward[i] = 0.0
        if terminal[i] and \
                reward_signal_updated[i] == 0 and \
                agent.state == TrainState.DONE and \
                env.raw_env._elapsed_steps < (env.raw_env._max_episode_steps - 5):
            reward[i] = 1.0
            reward_signal_updated[i] = 1.0
        elif reward_signal_updated[i] == 0.0 and terminal[i]:
            reward[i] = -1.0
            reward_signal_updated[i] = 1.0
        else:
            reward[i] -= 0.001

    return reward




policy_creator_list: List[Callable[[int, int], Policy]] = [
                                                           create_random_policy,     #0
                                                           create_td3_policy,        #1
                                                           create_a2c_policy,        #2
                                                           create_ppo_policy,        #3
                                                           create_dddqn_policy,      #4
                                                           create_dp_ppo_policy,     #5
                                                           create_dp_ppo_policy_dla, #6
                                                           create_ppo_policy_dla     #7
                                                           ]


test_with_deadlock_avoidance_policy = True

if __name__ == "__main__":
    do_rendering = False
    do_training = True

    environment = RailEnvironmentPersistable(
        obs_builder_object_creator=create_obs_builder_object,
        grid_width=30,
        grid_height=40,
        grid_mode=True,
        number_of_agents=10)
    environment.generate_and_persist_environments(generate_nbr_env=10,
                                                  generate_agents_per_env=[1, 2, 3, 4, 5],  # [1, 2, 3, 5, 10, 20, 30, 50],
                                                  overwrite_existing=False)
    environment.load_environments_from_path()

    if test_with_deadlock_avoidance_policy:
        solver_deadlock = FlatlandSolver(environment,
                                         create_deadlock_avoidance_policy(environment, environment.get_action_space()),
                                         FlatlandSimpleRenderer(environment) if do_rendering else None)
        solver_deadlock.perform_training(max_episodes=1000)

    else:
        if do_training:
            for pcl in policy_creator_list:
                solver = FlatlandSolver(environment,
                                        pcl(environment.get_observation_space(), environment.get_action_space()),
                                        FlatlandSimpleRenderer(environment) if do_rendering else None)
                solver.set_reward_shaper(flatland_reward_shaper)
                # solver.load_policy()
                solver.perform_training(max_episodes=1000)

