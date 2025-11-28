from typing import Any, Callable, Optional, Type, List, Union, Tuple, Set, Dict
from collections import namedtuple, deque

import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.core.grid.grid4_utils import get_new_position

from flatland.envs.agent_utils import EnvAgent
from flatland.envs.step_utils.states import TrainState
from flatland_railway_extension.RailroadSwitchAnalyser import RailroadSwitchAnalyser
from flatland.envs.fast_methods import fast_count_nonzero, fast_argmax, fast_position_equal

from environment.environment import Environment
from example.flatland_rail_env.flatland_rail_env_persister import RailEnvironmentPersistable
from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.deadlock_avoidance_policy import \
    DeadLockAvoidancePolicy
from policy.learning_policy.learning_policy import LearningPolicy
from rendering.flatland.flatland_simple_renderer import FlatlandSimpleRenderer
from solver.flatland.flatland_solver import FlatlandSolver
from solver.multi_agent_base_solver import RewardList, TerminalList, InfoDict
from policy.policy import Policy

from utils.flatland.shortest_distance_walker import ShortestDistanceWalker
from utils.training_evaluation_pipeline import create_random_policy

from marl_attention_temporal_mappo import MARL_ATTENTION_TEMPORAL_PPOPolicy, MARL_ATTENTION_TEMPORAL_MAPPO_Param

# Enforce disable GPU
import torch
torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# REUSE: WalkToNextDecisionPoint  
# =============================================================================

class WalkToNextDecisionPoint(ShortestDistanceWalker):
    def __init__(self, env: RailEnv):
        super(WalkToNextDecisionPoint, self).__init__(env)
        self.clear(None)
        self.switchAnalyser: RailroadSwitchAnalyser  = RailroadSwitchAnalyser(env) 

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


    def callback(self, handle, agent, position, direction, action, possible_transitions) -> bool:
        if self.final_pos is None:
            self.final_pos = position
            self.final_dir = direction
        self.target_found = int(fast_position_equal(agent.target, position))
        if self.target_found == 1:
            return False

        agent_at_railroad_switch, agent_near_to_railroad_switch, \
            agent_at_railroad_switch_cell, agent_near_to_railroad_switch_cell = \
            self.switchAnalyser.check_agent_decision(position=position, direction=direction)


        self.visited.append((position, direction))
        self.path.append(position)

        if agent_at_railroad_switch_cell:
            self.switch.append((position, direction))
        if agent_at_railroad_switch:
            self.dir_switch.append((position, direction))
        if agent_near_to_railroad_switch_cell:
            self.near_to_switch.append((position, direction))
        if agent_near_to_railroad_switch:
            self.near_to_dir_switch.append((position, direction))

        agent_idx = self.agent_map[position]
        if agent_idx != -1 and agent_idx != handle:
            self.other_agent_handles.append(agent_idx)
            self.dir_at_agent_handles.append(direction)

        self.final_pos = position
        self.final_dir = direction
        return True


OtherAgentsObservationInfo = namedtuple('OtherAgentsObservationInfo',
                                        ['handle',
                                         'same_direction',
                                         'opp_direction',
                                         'detected_branch_action'])


# =============================================================================
# BASE OBSERVATION: ExperimentalObservation  
# =============================================================================

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
        self.walker: WalkToNextDecisionPoint  = None

        self.observation_space = np.zeros(
            ExperimentalObservation.getObservationSize() - ExperimentalObservation.getObservationOthersExtraSize(),
            dtype=np.float32
        )

    @staticmethod
    def getObservationOthersExtraSize() -> int:
        return 7  # 3 for self presence + 3 for self direction + 1 padding

    @staticmethod
    def getObservationSize() -> int:
        # 7 agent_state + 4 switch + 18 transition (3×6) + 1 target = 30
        return 37 + ExperimentalObservation.getObservationOthersExtraSize()

    def reset(self):
        # Initialize analyzers
        self.switchAnalyser = RailroadSwitchAnalyser(self.env)
        self.walker = WalkToNextDecisionPoint(self.env)
        
    @staticmethod
    def get_pos_dir(agent: EnvAgent):
        pos = agent.position if agent.position is not None else agent.initial_position
        dir = agent.direction if agent.direction is not None else agent.initial_direction
        return pos, dir

    def get_many(self, handles: Optional[List[int]] = None) -> Any:
       
        # Initialize analyzers
        #self.switchAnalyser = RailroadSwitchAnalyser(self.env)
        # self.walker = WalkToNextDecisionPoint(self.env)

        # Build agent map
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

        # Prepare multi-head attention format
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

                    # One-hot encoding for detection branch
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

    def get(self, handle: int = 0):

        distance_map = self.env.distance_map.get()
        v = distance_map[handle]
        max_dist = np.max(v[v != np.inf])

        agent = self.env.agents[handle]
        pos, dir = self.get_pos_dir(agent)
        target = agent.target

        if pos is None or target is None:
            vec_len = (ExperimentalObservation.getObservationSize() -
                      ExperimentalObservation.getObservationOthersExtraSize())
            return (np.zeros(vec_len, dtype=np.float32)-1, [])

        vec = []
        opp_agents = set()
 
        agent_at_railroad_switch, agent_near_to_railroad_switch, \
            agent_at_railroad_switch_cell, agent_near_to_railroad_switch_cell = \
            self.switchAnalyser.check_agent_decision(position=pos, direction=dir)

        dist = distance_map[handle, pos[0], pos[1], dir]

        # Agent State Features [0-6]
        vec.append(1.0 if agent.state == TrainState.WAITING else -1.0)
        vec.append(1.0 if agent.state == TrainState.READY_TO_DEPART else -1.0)
        vec.append(1.0 if agent.state == TrainState.MALFUNCTION_OFF_MAP else -1.0)
        vec.append(1.0 if agent.state == TrainState.MOVING else -1.0)
        vec.append(1.0 if agent.state == TrainState.STOPPED else -1.0)
        vec.append(1.0 if agent.state == TrainState.MALFUNCTION else -1.0)
        vec.append(1.0 if agent.state == TrainState.DONE else -1.0)

        # Railroad Switch Features [7-10]
        vec.append(agent_at_railroad_switch * 2.0 - 1.0)
        vec.append(agent_near_to_railroad_switch * 2.0 - 1.0)
        vec.append(agent_at_railroad_switch_cell * 2.0 - 1.0)
        vec.append(agent_near_to_railroad_switch_cell * 2.0 - 1.0)

        # Walk forward to next switch - if there is a an agent (opposite/other) in this segment
        # dead lock avoidance is too late 
        new_position = pos
        new_direction = dir
        visited = [pos]
        transitions = self.env.rail.get_transitions(*new_position, new_direction)
        has_other_agent_in_segment = False
        has_other_agent_in_segment_with_opposite_dir = False    
        while fast_count_nonzero(transitions) == 1 and new_position != target:
            new_position, new_direction, _1, _2, _3 = \
                self.walker.walk(handle, new_position, new_direction)
            if self.agent_map[new_position] != -1 and self.agent_map[new_position] != handle:
                has_other_agent_in_segment = True   
                if self.env.agents[self.agent_map[new_position]].direction != new_direction:
                    has_other_agent_in_segment_with_opposite_dir = True
            transitions = self.env.rail.get_transitions(*new_position, new_direction)
            visited.append(new_position)
        
        vec.append(int(has_other_agent_in_segment) * 2.0 - 1.0)
        vec.append(int(has_other_agent_in_segment_with_opposite_dir) * 2.0 - 1.0)
        vec.append(int(new_position != target) * 2.0 - 1.0)


        dir = new_direction
        pos = new_position
        # Direction Analysis [11-28]: LEFT, FORWARD, RIGHT
        for action_i in range(-1, 2):  # -1=left, 0=forward, 1=right
            new_direction = (dir + action_i) % 4
            if transitions[new_direction]:
                npos = get_new_position(pos, new_direction)

                self.walker.clear(self.agent_map)
                self.walker.walk_to_target(handle, npos, new_direction, max_step=50)

                for d in self.walker.path:
                    visited.append(d)

                other_agents_cnt_opp_dir = 0
                for idx in range(len(self.walker.other_agent_handles)):
                    other_agent_h = self.walker.other_agent_handles[idx]
                    other_dir = self.env.agents[other_agent_h].direction
                    my_dir = self.walker.dir_at_agent_handles[idx]

                    opp_agents.add(
                        OtherAgentsObservationInfo(
                            handle=other_agent_h,
                            same_direction=(other_dir == my_dir),
                            opp_direction=(other_dir != my_dir),
                            detected_branch_action=(action_i + 1)
                        )
                    )
                    if other_dir != my_dir:
                        other_agents_cnt_opp_dir += 1

                len_path = len(self.walker.visited)
                if len_path == 0:
                    len_path = 1

                # 6 features per direction 
                vec.append(len(self.walker.switch) - len(self.walker.dir_switch))
                vec.append(len(self.walker.near_to_switch) - len(self.walker.near_to_switch))
                vec.append(1.0 - len(self.walker.switch) / len_path)
                vec.append(1.0 - len(self.walker.dir_switch) / len_path)
                vec.append(1.0 - len(self.walker.near_to_switch) / len_path)
                vec.append(1.0 - len(self.walker.near_to_dir_switch) / len_path)
                vec.append(1.0 - len(self.walker.switch) / len_path)
                same_dir_agents = len(self.walker.other_agent_handles) - other_agents_cnt_opp_dir
                vec.append(1.0 - same_dir_agents / len_path)
                vec.append(1.0 - other_agents_cnt_opp_dir / len_path)
                vec.append(1.0 if target in self.walker.path else -1.0)

                new_dist = distance_map[handle, npos[0], npos[1], new_direction]
                if new_dist == np.inf or dist == np.inf:
                    vec.append(-1.0)
                    vec.append(-1.0)
                else:
                    on_optimal_path = 1.0 if new_dist < dist else 0.0
                    vec.append(on_optimal_path)
                    vec.append(1.0 - new_dist / (1.0 + dist))

            else:
                vec.append(-1.0)
                vec.append(-1.0)
                vec.append(-1.0)
                vec.append(-1.0)
                vec.append(-1.0)
                vec.append(-1.0)

        # Target Status [29]
        vec.append(1.0 if pos == target else -1.0)

        arr = np.array(vec, dtype=np.float32)
        vec_len = (ExperimentalObservation.getObservationSize() -
                  ExperimentalObservation.getObservationOthersExtraSize())
        if arr.shape[0] != vec_len:
            if arr.shape[0] < vec_len:
                pad = np.zeros(vec_len - arr.shape[0], dtype=np.float32)
                arr = np.concatenate([arr, pad])
            else:
                arr = arr[:vec_len]

        self.env.dev_obs_dict.update({handle: visited})

        return (arr, opp_agents)


# =============================================================================
# NEW: TemporalMultiAgentObservation - Adds Temporal Dimension!
# =============================================================================

class TemporalMultiAgentObservation(ObservationBuilder):
    """
    🚀 INNOVATION: Temporal Observation Builder
    
    Extends ExperimentalObservation with:
    1. Temporal Buffer: Stores last T timesteps (default T=3)
    2. Velocity Features: Computed from position/direction deltas
    3. Sequential Format: Returns [(obs_t-2, opp_t-2), (obs_t-1, opp_t-1), (obs_t, opp_t)]
    
    Observation Size: 33D per timestep
    - 30D: Base features (from ExperimentalObservation)
    - 3D:  Velocity (velocity_x, velocity_y, angular_velocity)
    """
    
    def __init__(self, temporal_window: int = 3):
        super().__init__()
        self.temporal_window = temporal_window
        self.base_obs = ExperimentalObservation()
        self.env = None  # Will be set by set_env()
        
        # Temporal history per agent: {handle: deque([obs_t-2, obs_t-1, obs_t])}
        self.temporal_history: Dict[int, deque] = {}
        
        # Last positions for velocity computation: {handle: (pos, dir)}
        self.last_positions: Dict[int, Tuple] = {}
        
    @staticmethod
    def getObservationSize() -> int:
        # Base 30D + Velocity 3D = 33D per timestep
        return 30 + 3
    
    def set_env(self, env):
        """Set environment reference and propagate to base_obs"""
        super().set_env(env)
        self.env = env
        self.base_obs.set_env(env)
    
    def reset(self):
        """Reset temporal buffers at episode start"""
        self.base_obs.reset()
        self.temporal_history = {}
        self.last_positions = {}
        self.base_obs.reset()
    
    def get_many(self, handles: Optional[List[int]] = None):


        """
        Returns temporal sequences for all agents
        
        Output format: List of temporal_sequences
        temporal_sequence = [(obs_t-2, opp_t-2), (obs_t-1, opp_t-1), (obs_t, opp_t)]
        obs_t = 33D numpy array (30 base + 3 velocity)
        opp_t = list of 33D arrays
        """
        if handles is None:
            handles = list(range(len(self.env.agents)))
        # Get current observations from base builder
        current_obs = self.base_obs.get_many(handles)
        
        # Enrich with velocity features
        enriched_obs = []
        for handle_idx, (obs_self, obs_others) in enumerate(current_obs):
            agent = self.env.agents[handle_idx]
            
            # Compute velocity from position delta
            velocity_features = self._compute_velocity(handle_idx, agent)
            
            # Combine: [base_obs (30D) | velocity (3D)] = 33D
            # obs_self is already 37D (30 base + 7 padding), take only first 30D
            obs_base = obs_self[:30]
            obs_enriched = np.concatenate([obs_base, velocity_features])
            
            # Also enrich opponent observations
            obs_others_enriched = []
            for opp_obs in obs_others:
                # opp_obs is 37D, take first 30D and add velocity
                opp_base = opp_obs[:30]
                # For opponents, we don't have velocity history, so use zeros
                opp_vel = np.zeros(3, dtype=np.float32)
                opp_enriched = np.concatenate([opp_base, opp_vel])
                obs_others_enriched.append(opp_enriched)
            
            enriched_obs.append((obs_enriched, obs_others_enriched))
        
        # Build temporal sequences
        temporal_sequences = []
        for handle_idx, (obs_self, obs_others) in enumerate(enriched_obs):
            # Initialize buffer if needed
            if handle_idx not in self.temporal_history:
                self.temporal_history[handle_idx] = deque(maxlen=self.temporal_window)
            
            # Add current observation
            self.temporal_history[handle_idx].append((obs_self, obs_others))
            
            # Get sequence with padding
            seq = list(self.temporal_history[handle_idx])
            
            # Padding: Duplicate first observation if not enough history
            while len(seq) < self.temporal_window:
                if len(seq) > 0:
                    seq.insert(0, seq[0])
                else:
                    # Very first observation - create zero observation
                    zero_obs = np.zeros(33, dtype=np.float32)
                    seq.insert(0, (zero_obs, []))
            
            temporal_sequences.append(seq)
        
        return temporal_sequences
    
    def _compute_velocity(self, handle: int, agent) -> np.ndarray:
        """
        Compute velocity from position/direction deltas - OPTIMIZED
        
        Returns:
            [move_forward_dist_based, at_station_entry, delta_trans] (3D)
        """
        # ⚡ OPTIMIZATION: Pre-allocate result
        velocity = np.zeros(3, dtype=np.float32)
        
        current_pos, current_dir = self.base_obs.get_pos_dir(agent)
        
        if handle not in self.last_positions or current_pos is None:
            # First observation or agent not on map
            velocity[0] = -1.0
            velocity[1] = -1.0
            self.last_positions[handle] = (current_pos, current_dir)
            return velocity
        
        last_pos, last_dir = self.last_positions[handle]

        # Position delta (normalized to grid size)
        if last_pos is not None and current_pos is not None:
            # ⚡ OPTIMIZATION: Cache distance_map lookup
            distance_map = self.env.distance_map.get()
            cur_dist = distance_map[handle, current_pos[0], current_pos[1], current_dir]
            last_dist = distance_map[handle, last_pos[0], last_pos[1], last_dir]

            # ⚡ OPTIMIZATION: Avoid repeated rail.get_transitions calls
            cur_transitions = self.env.rail.get_transitions(*current_pos, current_dir)
            last_transitions = self.env.rail.get_transitions(*last_pos, last_dir)
            cur_options = fast_count_nonzero(cur_transitions)  
            last_options = fast_count_nonzero(last_transitions)  
            
            velocity[1] = float(cur_options + last_options)  # at_station_entry
            velocity[2] = float(cur_options - last_options)  # delta_trans

            if cur_dist == np.inf or last_dist == np.inf:
                velocity[0] = -1.0
            else:
                delta_dist = (cur_dist - last_dist) / (1 + cur_dist)
                velocity[0] = delta_dist
        else:
            velocity[0] = -1.0
            velocity[1] = -1.0
            # velocity[2] = 0.0  # already initialized
        
        # Update history
        self.last_positions[handle] = (current_pos, current_dir)
        
        return velocity





class MARL_ATT_DecisionPointPolicy(MARL_ATTENTION_TEMPORAL_PPOPolicy):
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 in_parameters: Union[MARL_ATTENTION_TEMPORAL_MAPPO_Param, None] = None, 
                 show_pre_train_debug_msg=False,
                 show_progress_bar=True,
                 train_frequency=10,
                 use_deadlock_avoidance_policy = False):
        self.deadlock_avoidance_policy = None
        self.use_deadlock_avoidance_policy = use_deadlock_avoidance_policy
        super(MARL_ATT_DecisionPointPolicy, self).__init__(
                state_size,
                action_size,
                in_parameters,
                show_pre_train_debug_msg,
                show_progress_bar,
                train_frequency
            )
        self._env: Union[Environment, None] = None
        self.switchAnalyser: Union[RailroadSwitchAnalyser, None] = None

    def get_name(self):
        if self.use_deadlock_avoidance_policy:
            return self.__class__.__name__ + "_DLA"
        return self.__class__.__name__


    def step(self, handle, state, action, reward, next_state, done):
        super(MARL_ATT_DecisionPointPolicy, self).step(handle, state, action, reward, next_state, done)
        if self.use_deadlock_avoidance_policy:
            if self.deadlock_avoidance_policy is not None:
                self.deadlock_avoidance_policy.step(handle, state, action, reward, next_state, done)

    def start_step(self, train):
        super(MARL_ATT_DecisionPointPolicy, self).start_step(train)
        if self.use_deadlock_avoidance_policy:
            self.deadlock_avoidance_policy.start_step(train)

    def reset(self, env: Environment):
        self._env = env
        self.switchAnalyser = RailroadSwitchAnalyser(env.raw_env)
        super(MARL_ATT_DecisionPointPolicy, self).reset(env)
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
        if agent.state == TrainState.MOVING:
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

        action = super(MARL_ATT_DecisionPointPolicy, self).act(handle, state, eps)
        if self.use_deadlock_avoidance_policy:
            if agent.state.is_on_map_state():
                if action == RailEnvActions.DO_NOTHING:
                    dla_action = self.deadlock_avoidance_policy.act(handle, state, eps)
                    return dla_action
                    # if RailEnvActions.STOP_MOVING == dla_action:
                    #     return RailEnvActions.STOP_MOVING

        return action


# =============================================================================
# ENVIRONMENT & TRAINING SETUP
# =============================================================================

def create_temporal_obs_builder_object():
    """Factory for TemporalMultiAgentObservation"""
    return TemporalMultiAgentObservation(temporal_window=3)


ppo_param = MARL_ATTENTION_TEMPORAL_MAPPO_Param(
    hidden_size=128,
    batch_size=512,  # ⚡ Increased for stable convergence to 100%
    learning_rate=2e-4,  # ⚡ Reduced for fine-tuning to perfection
    discount=0.995,
    use_gpu=True,
    max_episodes_in_training_memory=40,  # ⚡ Store more episodes for full coverage
    batch_fraction=0.1,
    k_epochs=3,  # ⚡ More epochs = better memorization
    max_batches_per_training=15,
    temporal_window=3  # NEW: Temporal dimension
)

def create_ma_ppo_agent(observation_space: int, action_space: int) -> LearningPolicy:
    """
    Creates PPO Policy with Temporal Transformer Encoder
    
    observation_space: 33 (30 base + 3 velocity)
    """
    print('>> MARL_ATTENTION_TEMPORAL_PPOPolicy (Temporal Transformer)')
    print('   - observation_space:', observation_space)
    print('   - action_space:', action_space)
    print('   - temporal_window: 3 timesteps')
    print('   - architecture: 2-Level Attention (Temporal + Spatial)')
        
    return MARL_ATTENTION_TEMPORAL_PPOPolicy(
        observation_space,
        action_space,
        ppo_param,
        show_pre_train_debug_msg=False,
        show_progress_bar=True,
        train_frequency=10
    )

def create_ma_ppo_agent_dp(observation_space: int, action_space: int) -> LearningPolicy:
    """
    Creates  PPO Policy with Temporal Transformer Encoder
    
    observation_space: 33 (30 base + 3 velocity)
    """
    print('>> MARL_ATT_DecisionPointPolicy (Temporal Transformer)')
    print('   - observation_space:', observation_space)
    print('   - action_space:', action_space)
    print('   - temporal_window: 3 timesteps')
    print('   - architecture: 2-Level Attention (Temporal + Spatial)')
        
    return MARL_ATT_DecisionPointPolicy(
        observation_space,
        action_space,
        ppo_param,
        show_pre_train_debug_msg=False,
        show_progress_bar=True,
        train_frequency=10,
        use_deadlock_avoidance_policy=False
    )

 
def create_ma_ppo_agent_dp_DLA(observation_space: int, action_space: int) -> LearningPolicy:
    """
    Creates PPO Policy with Temporal Transformer Encoder
    
    observation_space: 33 (30 base + 3 velocity)
    """
    print('>> MARL_ATT_DecisionPointPolicy with Deadlockavoidance (Temporal Transformer)')
    print('   - observation_space:', observation_space)
    print('   - action_space:', action_space)
    print('   - temporal_window: 3 timesteps')
    print('   - architecture: 2-Level Attention (Temporal + Spatial)')
    
    return MARL_ATT_DecisionPointPolicy(
        observation_space,
        action_space,
        ppo_param,
        show_pre_train_debug_msg=False,
        show_progress_bar=True,
        train_frequency=10,
        use_deadlock_avoidance_policy=True
    )

def create_deadlock_avoidance_policy(environment: Environment,
                                     action_space: int,
                                     show_debug_plot=False) -> DeadLockAvoidancePolicy:
    return DeadLockAvoidancePolicy(environment.get_raw_env(),
                                   action_space,
                                   enable_eps=False,
                                   show_debug_plot=show_debug_plot)



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

        pos, dir = ExperimentalObservation.get_pos_dir(agent)
        dist = distance_map[i, pos[0], pos[1], dir]
        max_dist = np.max(distance_map[i][distance_map[i] != np.inf]) + 1
        if max_dist == np.inf:
            max_dist = 1.0
            dist = 1.0
        if dist == np.inf:
            dist = max_dist
        reward[i] = -dist / max_dist
        if terminal[i] and \
                reward_signal_updated[i] == 0 and \
                agent.state == TrainState.DONE and \
                env.raw_env._elapsed_steps < (env.raw_env._max_episode_steps - 5):
            reward[i] = 1.0
            reward_signal_updated[i] = 1.0
        elif reward_signal_updated[i] == 0.0 and terminal[i]:
            reward[i] = 0.01
            reward_signal_updated[i] = 1.0
        else:
            reward[i] -= 0.001

    return reward



policy_creator_list: List[Callable[[int, int], Policy]] = [
    # create_random_policy, 
    create_ma_ppo_agent,
    # create_ma_ppo_agent_dp,
    # create_ma_ppo_agent_dp_DLA
]


if __name__ == "__main__":
    do_rendering = False
    do_training = True
    test_with_deadlock_avoidance_policy = False

    print("\n" + "="*80)
    print("🚀 Temporal Multi-Agent Transformer")
    print("="*80)
    print("Innovations:")
    print("  ✅ Temporal Observation Buffer (3 timesteps)")
    print("  ✅ Velocity Features (dx, dy, angular_vel)")
    print("  ✅ 2-Level Transformer (Temporal + Spatial Attention)")
    print("="*80 + "\n")

    environment = RailEnvironmentPersistable(
        obs_builder_object_creator=create_temporal_obs_builder_object,
        grid_width=30,
        grid_height=40,
        grid_mode=True,
        number_of_agents=10
    )
    
    environment.generate_and_persist_environments(
        generate_nbr_env=10,
        generate_agents_per_env=[1, 2, 5, 10], 
        overwrite_existing=False
    )
    environment.load_environments_from_path()

    if test_with_deadlock_avoidance_policy:
        solver_deadlock = FlatlandSolver(
            environment,
            create_deadlock_avoidance_policy(environment, environment.get_action_space()),
            FlatlandSimpleRenderer(environment) if do_rendering else None
        )
        if do_training:
            solver_deadlock.perform_training(max_episodes=1000)
        else:
            solver_deadlock.perform_evaluation(max_episodes=1000)
    else:
        for pcl in policy_creator_list:
            # Pass 33D observation size (30 base + 3 velocity)
            policy = pcl(
                TemporalMultiAgentObservation.getObservationSize(),
                environment.get_action_space()
            )
            
            if hasattr(policy, 'get_training_summary'):
                policy.get_training_summary()
            
            solver = FlatlandSolver(
                environment,
                policy,
                FlatlandSimpleRenderer(environment) if do_rendering else None
            )
            solver.set_reward_shaper(flatland_reward_shaper)
            if do_training:
                solver.load_policy()  # Uncomment to continue training
                solver.perform_training(max_episodes=1000)
            else:
                solver.load_policy()   
                solver.perform_evaluation(max_episodes=1000)
