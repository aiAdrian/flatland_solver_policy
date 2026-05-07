# =============================================================================
# References used in this file (directed-maze MARL on Flatland-RL,
# deadlock avoidance, action masking, safety shielding).
# -----------------------------------------------------------------------------
# [1] Mohanty et al. (2020). "Flatland-RL: Multi-Agent Reinforcement Learning
#     on Trains." arXiv:2012.05893.  https://arxiv.org/abs/2012.05893
# [2] Laurent, Schneider, Scheller, et al. (2021). "Flatland Competition 2020:
#     MAPF and MARL for Efficient Train Coordination on a Grid World."
#     arXiv:2103.16511.  https://arxiv.org/abs/2103.16511
#     -- Top submissions combine RL with deadlock-avoidance heuristics and
#     decision-point reductions; see Sec. 4 ("MARL approaches").
# [3] Yu et al. (2022). "The Surprising Effectiveness of PPO in Cooperative
#     Multi-Agent Games" (MAPPO). arXiv:2103.01955.
#     https://arxiv.org/abs/2103.01955
# [4] Huang & Ontañón (2022). "A Closer Look at Invalid Action Masking in
#     Policy Gradient Algorithms." arXiv:2006.14171.
#     https://arxiv.org/abs/2006.14171
#     -- Justifies masking invalid (off-rail / deadlock-bound) actions BEFORE
#     the categorical sample, not after.
# [5] Alshiekh et al. (2018). "Safe Reinforcement Learning via Shielding."
#     AAAI. arXiv:1708.08611.  https://arxiv.org/abs/1708.08611
#     -- Theoretical basis for using DeadLockAvoidancePolicy as a *shield*
#     that overrides unsafe RL actions.
# [6] Sartoretti et al. (2019). "PRIMAL: Pathfinding via Reinforcement and
#     Imitation Multi-Agent Learning." IEEE RA-L. arXiv:1809.03531.
#     https://arxiv.org/abs/1809.03531
#     -- Decentralized partially-observable MARL on grid mazes; combines
#     learned policy with a centralized expert (here: DLA) for deadlock-free
#     execution.
# [7] Ng, Harada, Russell (1999). "Policy Invariance Under Reward
#     Transformations" (PBRS). ICML.
# =============================================================================

from typing import Any, Callable, Optional, Type, List, Union, Tuple, Set, Dict
from collections import namedtuple, deque
import os
import numpy as np
import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.step_utils.states import TrainState
from flatland_railway_extension.RailroadSwitchAnalyser import RailroadSwitchAnalyser
from flatland.envs.fast_methods import fast_count_nonzero, fast_argmax, fast_position_equal
from environment.environment import Environment
from example.flatland_rail_env.flatland_rail_env_persister import RailEnvironmentPersistable
from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.deadlock_avoidance_policy import DeadLockAvoidancePolicy
from policy.learning_policy.learning_policy import LearningPolicy
from rendering.flatland.flatland_simple_renderer import FlatlandSimpleRenderer
from solver.flatland.flatland_solver import FlatlandSolver
from solver.multi_agent_base_solver import RewardList, TerminalList, InfoDict
from policy.policy import Policy
from utils.flatland.shortest_distance_walker import ShortestDistanceWalker
from utils.training_evaluation_pipeline import create_random_policy
from marl_attention_temporal_mappo import MARL_ATTENTION_TEMPORAL_PPOPolicy, MARL_ATTENTION_TEMPORAL_MAPPO_Param
from marl_attention_temporal_observation.experimental_observation import ExperimentalObservation
from marl_attention_temporal_observation.temporal_multi_agent_observation import TemporalMultiAgentObservation
from marl_attention_temporal_observation.hierarchical_routes_observation import HierarchicalRoutesObservation
from decider_policy import DeciderPPOPolicy
from marl_attention_temporal_observation.simplified_path_three_tier_observation import SimplifiedPathThreeTierObservation

# Enforce disable GPU
torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    # ------------------------------------------------------------------
    # Invalid-Action Masking for Flatland's directed maze
    # ------------------------------------------------------------------
    # Refs: Huang & Ontañón (2022) arXiv:2006.14171 -- masking invalid
    # logits before the categorical sample is provably equivalent to a
    # valid policy-gradient on the masked MDP (no off-policy bias).
    # In Flatland the rail graph is directed (cells have allowed
    # transitions per heading); illegal actions waste exploration budget
    # and produce immediate deadlocks at switches.
    # Action ids (flatland.envs.rail_env.RailEnvActions):
    #   0 DO_NOTHING, 1 MOVE_LEFT, 2 MOVE_FORWARD, 3 MOVE_RIGHT, 4 STOP_MOVING
    # ------------------------------------------------------------------
    def _legal_action_mask(self, agent: EnvAgent) -> np.ndarray:
        """Return a 0/1 mask of length action_size; 1 = legal."""
        mask = np.zeros(self.action_size, dtype=np.float32)
        # STOP_MOVING and DO_NOTHING are always legal -- they never derail.
        mask[RailEnvActions.DO_NOTHING] = 1.0
        mask[RailEnvActions.STOP_MOVING] = 1.0

        # If agent is not yet on the map, only DO_NOTHING / MOVE_FORWARD make
        # sense (MOVE_FORWARD triggers spawn). LEFT/RIGHT are not legal off-map.
        if not agent.state.is_on_map_state():
            mask[RailEnvActions.MOVE_FORWARD] = 1.0
            return mask

        position, direction = self._get_agent_position_and_direction(agent)
        if position is None:
            mask[RailEnvActions.MOVE_FORWARD] = 1.0
            return mask

        transitions = self._env.raw_env.rail.get_transitions(*position, direction)
        # Map (new_direction relative to current direction) -> RailEnvActions.
        # Flatland convention: forward = same direction;
        # left = (direction - 1) % 4; right = (direction + 1) % 4;
        # back = (direction + 2) % 4 (only legal at dead-ends).
        rel_to_action = {
            direction: RailEnvActions.MOVE_FORWARD,
            (direction - 1) % 4: RailEnvActions.MOVE_LEFT,
            (direction + 1) % 4: RailEnvActions.MOVE_RIGHT,
        }
        for new_dir in range(4):
            if transitions[new_dir] and new_dir in rel_to_action:
                mask[rel_to_action[new_dir]] = 1.0

        # Dead-end (only "back" allowed) -> Flatland encodes this as MOVE_FORWARD.
        if fast_count_nonzero(transitions) == 1 and \
                transitions[(direction + 2) % 4] == 1:
            mask[RailEnvActions.MOVE_FORWARD] = 1.0
        return mask

    def _masked_act(self, handle: int, state, eps: float) -> int:
        """Sample from the actor with invalid-action masking.

        Falls back to the unmasked parent implementation if the encoder
        is not yet available (defensive)."""
        agent: EnvAgent = self._env.raw_env.agents[handle]
        mask = self._legal_action_mask(agent)
        legal_actions = np.flatnonzero(mask > 0.5)
        if legal_actions.size == 0:
            return int(super(MARL_ATT_DecisionPointPolicy, self).act(handle, state, eps))
        # Make solver epsilon meaningful: random among legal rail actions.
        eps_val = float(eps) if eps is not None else 0.0
        # Keep a small exploration floor at decision points to avoid late-stage
        # deterministic lock-in at suboptimal switch behaviors.
        eps_floor = float(getattr(self, 'decision_eps_floor', 0.0))
        eps_val = max(eps_val, eps_floor)
        if eps_val > 0.0 and np.random.rand() < eps_val:
            return int(np.random.choice(legal_actions))
        try:
            with torch.no_grad():
                emb = self.encoder_actor.forward_agent(state, handle)
                logits = self.actor_critic_model.actor(emb.unsqueeze(0)).squeeze(0)
                mask_t = torch.from_numpy(mask).to(logits.device)
                # Standard masking (Huang & Ontañón 2022): set illegal logits
                # to a large negative number BEFORE softmax.
                logits = logits.masked_fill(mask_t < 0.5, -1e9)
                action = Categorical(logits=logits).sample().item()
            return int(action)
        except Exception:
            return int(np.random.choice(legal_actions))

    def act(self, handle: int, state, eps=0.):
        agent: EnvAgent = self._env.raw_env.agents[handle]
        position, direction = self._get_agent_position_and_direction(agent)
        agent_at_railroad_switch, agent_near_to_railroad_switch, \
            agent_at_railroad_switch_cell, agent_near_to_railroad_switch_cell = \
            self.switchAnalyser.check_agent_decision(position=position, direction=direction)

        # only if the agent is moving
        if agent.state.is_on_map_state():
            # Decision-point reduction (Laurent et al. 2021, arXiv:2103.16511,
            # Sec. 4): outside switches there is exactly one legal heading,
            # so we hard-code MOVE_FORWARD and let the RL policy only act at
            # genuine decision points. This drastically shortens credit
            # assignment paths and is what every top-5 solution did.
            if not agent_at_railroad_switch and not agent_near_to_railroad_switch_cell:
                return RailEnvActions.MOVE_FORWARD

        # Masked sampling at decision points.
        action = self._masked_act(handle, state, eps)

        # ------------------------------------------------------------
        # Safety shield (Alshiekh et al. 2018, arXiv:1708.08611):
        # the deadlock-avoidance heuristic [1, 2] is consulted as a
        # *shield* that may override the RL action whenever the heuristic
        # decides the agent must stop (head-on collision predicted along
        # shortest path). This implements the "RL + expert shield" pattern
        # used by the Flatland 2020 winners and PRIMAL (arXiv:1809.03531).
        # ------------------------------------------------------------
        if self.use_deadlock_avoidance_policy and self.deadlock_avoidance_policy is not None:
            if agent.state.is_on_map_state() or agent.state == TrainState.READY_TO_DEPART:
                dla_action = self.deadlock_avoidance_policy.act(handle, state, eps)
                # The DLA returns STOP_MOVING when it predicts the agent
                # cannot safely advance (no entry in agent_can_move). In
                # that case the shield wins -- preventing the deadlock.
                if dla_action == RailEnvActions.STOP_MOVING:
                    return RailEnvActions.STOP_MOVING
                # If the RL action is to enter a contested cell that DLA
                # rejects, fall back to DLA's safe action.
                if action == RailEnvActions.DO_NOTHING:
                    return dla_action
        return action


# =============================================================================
# ENVIRONMENT & TRAINING SETUP
# =============================================================================

# Globale Variable für die temporale Fenstergröße
TEMPORAL_WINDOW = 3  # 3 Frames -> Bewegung/Velocity wird durch Temporal-Attention nutzbar

# High-success curriculum: bias training toward hard coordination cases
# while keeping a small share of easy cases for stability.
PURE_MARL_AGENT_COUNTS = [5]
PURE_MARL_MAX_AGENTS = max(PURE_MARL_AGENT_COUNTS)
PURE_MARL_GRID_WIDTH = 30
PURE_MARL_GRID_HEIGHT = 40
PURE_MARL_N_CITIES = 3
PURE_MARL_NUM_ENVS = 18

# Curriculum with isolated scene caches per phase. This avoids mixing stale
# scenes and gives explicit control over increasing task complexity.
USE_CURRICULUM_PHASES = True
CURRICULUM_BASE_PATH = 'generated_envs'
# Targeting high done-ratio runs: keep phase5 focused on solvable coordination
# regimes by default. 5-agent traffic remains available as an optional stress
# test once policy quality is high and stable.
INCLUDE_5_AGENTS_IN_FINAL = False
# Pure MARL Curriculum: startet mit 1 Agent (reine Navigation, kein Koordinationsdruck),
# dann schrittweise Erhöhung. Erst wenn 1 Agent zuverlässig sein Ziel findet,
# macht Multi-Agent-Koordination Sinn.
CURRICULUM_PHASES = [
    {'name': 'phase0_nav',   'agent_counts': [1],          'num_envs': 10, 'episodes': 100},
    {'name': 'phase1_solo',  'agent_counts': [1, 2],        'num_envs': 12, 'episodes': 100},
    {'name': 'phase2_easy',  'agent_counts': [2, 3, 4],     'num_envs': 16, 'episodes': 100},
    {'name': 'phase3_mid',   'agent_counts': [3, 4, 5],     'num_envs': 18, 'episodes': 100},
    {'name': 'phase4_hard',  'agent_counts': [4, 5],        'num_envs': 22, 'episodes': 100},
    {'name': 'phase5_final', 'agent_counts': [5], 'num_envs': 50, 'episodes': 8000},
]

if INCLUDE_5_AGENTS_IN_FINAL:
    CURRICULUM_PHASES[-1]['agent_counts'] = [1, 2, 3, 4, 5]
    CURRICULUM_PHASES[-1]['episodes'] = 10000

# Toggle: when True, the temporal wrapper uses HierarchicalRoutesObservation
# (72D = 48 base + 24 sparse-neighbor block) as base. The decider policy expects
# this. The legacy 48D DecisionPointObservation works too, but the decider
# performs best with the extended layout.
USE_HIERARCHICAL_OBS = True

def create_temporal_obs_builder_object():
    """Factory for TemporalMultiAgentObservation"""
    if USE_HIERARCHICAL_OBS:
        base = HierarchicalRoutesObservation()
        return TemporalMultiAgentObservation(
            temporal_window=TEMPORAL_WINDOW,
            base_obs=base,
        )
    return TemporalMultiAgentObservation(temporal_window=TEMPORAL_WINDOW)


def create_decider_agent(observation_space: int, action_space: int) -> LearningPolicy:
    """Hierarchical Decider policy with Specialist sub-modules + PPO + 1-step
    deadlock BCE aux-loss. See HIERARCHICAL_DECIDER_ARCHITECTURE.md."""
    print('>> DeciderPPOPolicy (Hierarchical Specialists + Decider)')
    print('   - observation_space:', observation_space)
    print('   - action_space:', action_space)
    print('   - temporal_window:', TEMPORAL_WINDOW)
    return DeciderPPOPolicy(
        state_size=observation_space,
        action_size=action_space,
        learning_rate=5.0e-5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.10,
        k_epochs=1,
        batch_size=512,
        max_episodes_in_memory=20,
        # Conservative entropy pressure to keep exploration while reducing
        # destructive policy oscillations after BC warm-start.
        weight_entropy=0.04,
        weight_value=0.5,
        # Keep auxiliary signal active but avoid overpowering PPO objective.
        weight_aux_dl=0.035,
        temporal_window=TEMPORAL_WINDOW,
        train_frequency=10,     # ⬆️ Train every 10 episodes (faster feedback)
        reward_scale=0.005,
        aux_pos_weight=4.0,
        target_kl=0.04,
        max_eps_random=0.0,
        clear_buffer_after_update=True,
    )


ppo_param = MARL_ATTENTION_TEMPORAL_MAPPO_Param(
    hidden_size=64,         # ⬇️ Reduced for 4x faster LSTM (was 128)
    batch_size=256,
    learning_rate=1.5e-4,
    discount=0.99,  # Längere Belohnungsketten
    gae_lambda=0.97,  # Weniger Bias
    use_gpu=True,
    max_episodes_in_training_memory=20,   # ⬆️ Train after every 20 episodes
    k_epochs=3,
    batch_fraction=0.8,
    max_batches_per_training=10,          # ⬆️ Normal batch training (10 batches per update)
    temporal_window=TEMPORAL_WINDOW, # ⚡ MUST MATCH create_temporal_obs_builder_object()!
    encoder_type='lstm'
)

def create_ma_ppo_agent(observation_space: int, action_space: int) -> LearningPolicy:
    """
    Creates PPO Policy with Temporal Transformer Encoder
    
    observation_space: 33 (30 base + 3 velocity)
    """
    print('>> MARL_ATTENTION_TEMPORAL_PPOPolicy (Temporal Transformer)')
    print('   - observation_space:', observation_space)
    print('   - action_space:', action_space)
    print('   - temporal_window:', TEMPORAL_WINDOW)
    print('   - architecture: 2-Level Attention (Temporal + Spatial)')
        
    policy = MARL_ATTENTION_TEMPORAL_PPOPolicy(
        observation_space,
        action_space,
        ppo_param,
        show_pre_train_debug_msg=False,
        show_progress_bar=True,
        train_frequency=10    # ⬆️ Train every 10 episodes (faster feedback)
    )
    # Avoid late-stage over-conservative clipping that caused the 0.54-0.57 plateau.
    policy.surrogate_eps_clip = 0.15
    policy.weight_entropy = 0.012
    policy.stability_guard_start_episode = 2600
    policy.stability_guard_hard_episode = 3800
    return policy

def create_ma_ppo_agent_dp(observation_space: int, action_space: int) -> LearningPolicy:
    """
    Creates  PPO Policy with Temporal Transformer Encoder
    
    observation_space: 33 (30 base + 3 velocity)
    """
    print('>> MARL_ATT_DecisionPointPolicy (Temporal Transformer)')
    print('   - observation_space:', observation_space)
    print('   - action_space:', action_space)
    print('   - temporal_window:', TEMPORAL_WINDOW)
    print('   - architecture: 2-Level Attention (Temporal + Spatial)')
        
    policy = MARL_ATT_DecisionPointPolicy(
        observation_space,
        action_space,
        ppo_param,
        show_pre_train_debug_msg=False,
        show_progress_bar=True,
        train_frequency=10,   # ⬆️ Train every 10 episodes (faster feedback)
        use_deadlock_avoidance_policy=False
    )
    # Stable late-phase settings: keep learning without policy collapse.
    policy.surrogate_eps_clip = 0.16
    policy.weight_entropy = 0.065  # ⬆️ Increased entropy to encourage Left/Right
    policy.stability_guard_start_episode = 1200
    policy.stability_guard_hard_episode = 2600
    policy.ppo_target_kl = 0.07
    policy.ppo_max_kl = 0.14
    policy.ppo_emergency_kl = 0.24
    policy.ppo_emergency_kl_hard = 0.32
    policy.ratio_guard_soft = 1.12
    policy.ratio_guard_soft_low = 0.88
    policy.ratio_guard_hard = 1.28
    policy.ratio_guard_hard_low = 0.72
    policy.max_hard_batches_before_lr_decay = 10
    policy.hard_spike_streak_limit = 4
    policy.actor_lr_min_factor = 0.45
    policy.actor_lr_decay_on_instability = 0.85
    policy.max_eps_random = 0.10
    policy.decision_eps_floor = 0.01
    return policy

 
def create_ma_ppo_agent_dp_DLA(observation_space: int, action_space: int) -> LearningPolicy:
    """
    Creates PPO Policy with Temporal Transformer Encoder
    
    observation_space: 33 (30 base + 3 velocity)
    """
    print('>> MARL_ATT_DecisionPointPolicy with Deadlockavoidance (Temporal Transformer)')
    print('   - observation_space:', observation_space)
    print('   - action_space:', action_space)
    print('   - temporal_window:', TEMPORAL_WINDOW)
    print('   - architecture: 2-Level Attention (Temporal + Spatial)')
    
    policy = MARL_ATT_DecisionPointPolicy(
        observation_space,
        action_space,
        ppo_param,
        show_pre_train_debug_msg=False,
        show_progress_bar=True,
        train_frequency=10,   # ⬆️ Train every 10 episodes (faster feedback)
        use_deadlock_avoidance_policy=True
    )
    policy.surrogate_eps_clip = 0.15
    policy.weight_entropy = 0.030  # ⬆️ Increased entropy
    policy.stability_guard_start_episode = 2600
    policy.stability_guard_hard_episode = 3800
    # Same anti-stall settings for shielded training.
    policy.ppo_target_kl = 0.05
    policy.ppo_max_kl = 0.10
    policy.ppo_emergency_kl = 0.16
    policy.ppo_emergency_kl_hard = 0.24
    policy.ratio_guard_soft = 1.10
    policy.ratio_guard_soft_low = 0.90
    policy.ratio_guard_hard = 1.20
    policy.ratio_guard_hard_low = 0.80
    policy.max_hard_batches_before_lr_decay = 8
    policy.hard_spike_streak_limit = 3
    policy.actor_lr_min_factor = 0.35
    policy.actor_lr_decay_on_instability = 0.85
    policy.max_eps_random = 0.03
    policy.decision_eps_floor = 0.005
    return policy

def create_deadlock_avoidance_policy(environment: Environment,
                                     action_space: int,
                                     show_debug_plot=False) -> DeadLockAvoidancePolicy:
    return DeadLockAvoidancePolicy(environment.get_raw_env(),
                                   action_space,
                                   enable_eps=False,
                                   show_debug_plot=show_debug_plot)


# ============================================================================
# Reward Shaper: Ultra-Simple 4-Component System
# ============================================================================
# 1. Time cost: -0.01 per step (only on map)
# 2. Individual goal: +10 when agent reaches destination
# 3. Deadlock penalty: -20 for head-on collision (one-time)
# 4. Team success: +100 when all reach goals efficiently
# ============================================================================

class FlatlandPBRSShaper:
    """Minimal reward shaper: time cost + bonuses + deadlock penalty."""

    STEP_PENALTY = -0.01
    INDIVIDUAL_DONE_BONUS = 10.0
    ALL_DONE_BONUS = 100.0
    DEADLOCK_PENALTY = -20.0
    TIMEOUT_PENALTY = -8.0

    def __init__(self):
        self._done_charged: Dict[int, np.ndarray] = {}
        self._deadlock_charged: Dict[int, np.ndarray] = {}
        self._team_bonus_charged: Dict[int, bool] = {}
        self._team_fail_charged: Dict[int, bool] = {}
        self._prev_dist: Dict[int, np.ndarray] = {}
        self._diag: Dict[int, Dict[str, int]] = {}
        self.writer = None
        self.episode_count = 0
    
    def set_tensorboard_writer(self, writer):
        self.writer = writer

    def _is_local_deadlock(self, raw_env, agent) -> bool:
        """Head-on collision: agent blocked with opposite direction agent."""
        if agent.position is None:
            return False
        from flatland.envs.fast_methods import fast_count_nonzero
        transitions = raw_env.rail.get_transitions(*agent.position, agent.direction)
        if fast_count_nonzero(transitions) != 1:
            return False
        for ndir in range(4):
            if not transitions[ndir]:
                continue
            npos = get_new_position(agent.position, ndir)
            for other in raw_env.agents:
                if other.position == npos and (agent.direction) != other.direction:
                    other_t = raw_env.rail.get_transitions(*other.position, other.direction)
                    if fast_count_nonzero(other_t) == 1:
                        return True
        return False

    def __call__(self, reward, terminal, info, env):
        raw_env = env.raw_env
        agents = raw_env.agents
        num_agents = len(agents)
        env_id = id(raw_env)
        episode_done = bool(terminal.get('__all__', False)) if isinstance(terminal, dict) else False

        # Get current distances
        dm = raw_env.distance_map.get()
        distances = np.array([float(dm[a.handle, a.position[0], a.position[1], a.direction]) 
                              if a.position else float('inf') for a in agents], dtype=np.float32)

        # Initialize state if needed
        if env_id not in self._done_charged:
            self._done_charged[env_id] = np.zeros(num_agents, dtype=bool)
            self._deadlock_charged[env_id] = np.zeros(num_agents, dtype=bool)
            self._team_bonus_charged[env_id] = False
            self._team_fail_charged[env_id] = False
            self._prev_dist[env_id] = distances.copy()
            self._diag[env_id] = {'steps': 0, 'progress': 0, 'regress': 0, 'flat': 0}

        # Shaping per agent
        shaped = list(reward)
        for i, agent in enumerate(agents):
            s = 0.0
            if agent.state.is_on_map_state():
                s = self.STEP_PENALTY
            if agent.state == TrainState.DONE and not self._done_charged[env_id][i]:
                self._done_charged[env_id][i] = True
                s += self.INDIVIDUAL_DONE_BONUS
            if not self._deadlock_charged[env_id][i] and agent.position is not None and self._is_local_deadlock(raw_env, agent):
                self._deadlock_charged[env_id][i] = True
                s += self.DEADLOCK_PENALTY
            shaped[i] = float(reward[i] + s)

            # Track progress
            if agent.state.is_on_map_state() and np.isfinite(self._prev_dist[env_id][i]) and np.isfinite(distances[i]):
                self._diag[env_id]['steps'] += 1
                d = distances[i] - self._prev_dist[env_id][i]
                if d < -1e-6:
                    self._diag[env_id]['progress'] += 1
                elif d > 1e-6:
                    self._diag[env_id]['regress'] += 1
                else:
                    self._diag[env_id]['flat'] += 1

        # Team bonuses
        all_done = all(a.state == TrainState.DONE for a in agents)
        early = raw_env._elapsed_steps < raw_env._max_episode_steps - 10
        if episode_done and all_done and early and not self._team_bonus_charged[env_id]:
            self._team_bonus_charged[env_id] = True
            for i in range(num_agents):
                shaped[i] += self.ALL_DONE_BONUS
        if episode_done and not all_done and not early and not self._team_fail_charged[env_id]:
            self._team_fail_charged[env_id] = True
            for i, a in enumerate(agents):
                if a.state != TrainState.DONE:
                    shaped[i] += self.TIMEOUT_PENALTY

        # Log
        if episode_done:
            d = self._diag[env_id]
            n = max(1, d['steps'])
            if self.writer:
                self.writer.add_scalar("reward_shaper/progress", 100*d['progress']/n, self.episode_count)
                self.writer.add_scalar("reward_shaper/regress", 100*d['regress']/n, self.episode_count)
                self.writer.add_scalar("reward_shaper/flat", 100*d['flat']/n, self.episode_count)
                self.writer.add_scalar("reward_shaper/steps", d['steps'], self.episode_count)
            self.episode_count += 1
            self._diag[env_id] = {'steps': 0, 'progress': 0, 'regress': 0, 'flat': 0}

        self._prev_dist[env_id] = distances
        return shaped


flatland_reward_shaper = FlatlandPBRSShaper()


_policy_mode = os.environ.get('FLATLAND_POLICY_MODE', 'pure_marl_dla').strip().lower()
if _policy_mode == 'decider':
    policy_creator_list: List[Callable[[int, int], Policy]] = [
        create_decider_agent,
    ]
elif _policy_mode == 'pure_marl':
    policy_creator_list: List[Callable[[int, int], Policy]] = [
        create_ma_ppo_agent_dp,
    ]
else:
    # Default: shielded pure MARL for high completion and low deadlock risk.
    policy_creator_list: List[Callable[[int, int], Policy]] = [
        create_ma_ppo_agent_dp,#_DLA,
    ]


if __name__ == "__main__":
    import sys
    
    # Simple mode selection via command line
    # Usage: python marl_attention_temporal.py [phase5|continue|eval|dla_eval]
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else 'final'
    
    do_rendering = False
    checkpoint_interval = 100  # Default: every 100 episodes
    if mode == 'final':
        do_training = True
        test_with_deadlock_avoidance_policy = False
        start_from_phase = len(CURRICULUM_PHASES) - 1  # Only run last phase (dynamic index)
    elif mode == 'final_continue':
        do_training = True
        test_with_deadlock_avoidance_policy = False
        start_from_phase = len(CURRICULUM_PHASES) - 1  # Only run last phase (dynamic index)
        checkpoint_interval = 100  # Save checkpoint every 100 episodes for easy recovery
    elif mode == 'continue':
        do_training = True
        test_with_deadlock_avoidance_policy = False
        start_from_phase = 0  # Full curriculum
    elif mode == 'eval':
        do_training = False
        test_with_deadlock_avoidance_policy = False
    elif mode == 'dla_eval':
        do_training = True
        test_with_deadlock_avoidance_policy = True
    elif mode == 'dla':
        do_training = False
        test_with_deadlock_avoidance_policy = True
        start_from_phase = len(CURRICULUM_PHASES) - 1  # Only run last phase (dynamic index)
    else:
        print(f"Unknown mode '{mode}'. Choose: final, continue, eval, dla_eval")
        sys.exit(1)

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
        n_cities=PURE_MARL_N_CITIES,
        grid_width=PURE_MARL_GRID_WIDTH,
        grid_height=PURE_MARL_GRID_HEIGHT,
        grid_mode=True,
        number_of_agents=PURE_MARL_MAX_AGENTS,
        disable_mal_functions=True,
    )
    
    default_env_path = CURRICULUM_BASE_PATH
    # Only regenerate if missing (dont overwrite on each run for reproducibility)
    if (not os.path.exists(default_env_path)) or \
       len(os.listdir(default_env_path)) < PURE_MARL_NUM_ENVS:
        print(f"[Env] Generating {PURE_MARL_NUM_ENVS} environments...")
        environment.generate_and_persist_environments(
            generate_nbr_env=PURE_MARL_NUM_ENVS,
            generate_agents_per_env=PURE_MARL_AGENT_COUNTS,
            path=default_env_path,
            overwrite_existing=False  # Preserve existing
        )
    else:
        print(f"[Env] Using existing {default_env_path}")
    environment._loaded_env = []
    environment._loaded_env_itr = 0
    environment.load_environments_from_path(path=default_env_path)

    if test_with_deadlock_avoidance_policy:
        solver_deadlock = FlatlandSolver(
            environment,
            create_deadlock_avoidance_policy(environment, environment.get_action_space()),
            FlatlandSimpleRenderer(environment) if do_rendering else None
        )
        if do_training:
            solver_deadlock.perform_training(max_episodes=5000)
        else:
            solver_deadlock.perform_evaluation(max_episodes=1000)
    else:
        # Use the actual base-obs size so the policy gets a matching state_size
        # (48D for legacy DecisionPointObservation, 72D for HierarchicalRoutesObservation).
        _obs_builder_for_size = create_temporal_obs_builder_object()
        if hasattr(_obs_builder_for_size, 'get_observation_size'):
            _state_size = _obs_builder_for_size.get_observation_size()
        else:
            _state_size = TemporalMultiAgentObservation.getObservationSize()
        for pcl in policy_creator_list:
            policy = pcl(
                _state_size,
                environment.get_action_space()
            )
            if hasattr(policy, 'get_training_summary'):
                policy.get_training_summary()

            # ----------------------------------------------------------------
            # IL pre-train checkpoint loading (Behavior Cloning warm-start).
            # Activate via environment variable, e.g.
            #     IL_LOAD_CHECKPOINT=il_bc_checkpoint.pt python marl_attention_temporal.py
            # The checkpoint is produced by il_pretrain.py and contains the
            # DeciderNetwork.state_dict() trained on DeadLockAvoidancePolicy
            # demonstrations. Loading it before PPO bypasses the cold-start
            # plateau where credit-assignment with N>=4 agents is intractable
            # (Sartoretti et al. 2019, "PRIMAL"; arXiv:1809.03531).
            # ----------------------------------------------------------------
            import os as _os_il
            _il_ckpt = _os_il.environ.get('IL_LOAD_CHECKPOINT', '').strip()
            if not _il_ckpt:
                # Safe default: if a local BC checkpoint exists next to this script,
                # use it automatically to avoid accidental pure PPO cold-start runs.
                _candidate = _os_il.path.join(_os_il.path.dirname(__file__), 'il_bc_checkpoint.pt')
                if _os_il.path.exists(_candidate):
                    _il_ckpt = _candidate
                    print(f"\n[IL] Auto-detected BC checkpoint: {_il_ckpt}")

            if _il_ckpt and _os_il.path.exists(_il_ckpt) and hasattr(policy, 'load'):
                print(f"[IL] Loading BC pre-train checkpoint: {_il_ckpt}")
                policy.load(_il_ckpt)
                print("[IL] Checkpoint loaded — PPO will fine-tune from BC weights.")
            elif _il_ckpt:
                print(f"\n[IL] WARNING: IL_LOAD_CHECKPOINT={_il_ckpt!r} not found, "
                      f"falling back to random initialisation.")

            solver = FlatlandSolver(
                environment,
                policy,
                FlatlandSimpleRenderer(environment) if do_rendering else None
            )
            
            solver.set_reward_shaper(flatland_reward_shaper)
            if do_training:
                if mode == 'continue' or mode == 'final_continue':
                    if mode == 'final_continue':
                        # Load from training_output/last_checkpoint/ for automatic recovery
                        solver.load_policy(filename=f"training_output/last_checkpoint/{solver.get_name()}_{solver.policy.get_name()}")
                        print("✅ Loaded last checkpoint from training_output/last_checkpoint/")
                    else:
                        solver.load_policy()  # Load trained weights and continue
                if USE_CURRICULUM_PHASES:
                    phases_to_run = CURRICULUM_PHASES[start_from_phase:]
                    for phase in phases_to_run:
                        phase_path = f"{CURRICULUM_BASE_PATH}/{phase['name']}"
                        phase_agents = phase['agent_counts']
                        phase_num_envs = phase['num_envs']
                        phase_episodes = phase['episodes']

                        print("\n" + "=" * 80)
                        print(f"🎯 Curriculum {phase['name']}: agents={phase_agents}, envs={phase_num_envs}, episodes={phase_episodes}")
                        print("=" * 80)

                        if (not os.path.exists(phase_path)) or mode == 'continue':
                            environment.generate_and_persist_environments(
                                generate_nbr_env=phase_num_envs,
                                generate_agents_per_env=phase_agents,
                                path=phase_path,
                                overwrite_existing=False
                            )
                        environment._loaded_env = []
                        environment._loaded_env_itr = 0
                        environment.load_environments_from_path(path=phase_path)

                        print(f"[Train] {phase['name']}: {phase_episodes} episodes, agents={phase_agents}")
                        solver.perform_training(max_episodes=phase_episodes, checkpoint_interval=checkpoint_interval)
                else:
                    solver.perform_training(max_episodes=10000, checkpoint_interval=checkpoint_interval)
            else:
                solver.load_policy()   
                solver.perform_evaluation(max_episodes=1000)
