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
import sys
import argparse
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
                 use_deadlock_avoidance_policy=False,
                 optimizer_mode: str = 'single'):
        self.deadlock_avoidance_policy = None
        self.use_deadlock_avoidance_policy = use_deadlock_avoidance_policy
        super(MARL_ATT_DecisionPointPolicy, self).__init__(
                state_size,
                action_size,
                in_parameters,
                show_pre_train_debug_msg,
                show_progress_bar,
                train_frequency,
                optimizer_mode
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
    # Cell-Type Classification for State-Machine Reduction
    # ------------------------------------------------------------------
    # Classifies rail cells into 5 types to optimize decision-making:
    # OUTSIDE: Agent spawning (state type selection)
    # FORWARD_ONLY: Single rail path (no choice, hard-coded MOVE_FORWARD)
    # MERGING: Before a merge/switch (binary choice: forward or stop)
    # SWITCH: Multi-choice cell (left/forward/right)
    # DONE: Goal reached (no action needed)
    # ------------------------------------------------------------------
    def _classify_cell_type(self, agent: EnvAgent, raw_env) -> str:
        """Classify the current cell type of an agent.
        
        Returns: 'OUTSIDE' | 'FORWARD_ONLY' | 'MERGING' | 'SWITCH' | 'DONE'
        """
        # DONE state
        if agent.state == TrainState.DONE:
            return 'DONE'
        
        # OUTSIDE: not yet on map
        if agent.position is None or not agent.state.is_on_map_state():
            return 'OUTSIDE'
        
        # Get transitions at current position
        transitions = raw_env.rail.get_transitions(*agent.position, agent.direction)
        num_transitions = fast_count_nonzero(transitions)
        
        # SWITCH: >1 transition options
        if num_transitions > 1:
            return 'SWITCH'
        
        # Check next cell (one forward)
        try:
            next_pos = get_new_position(agent.position, agent.direction)
            next_dir = fast_argmax(transitions)
            next_transitions = raw_env.rail.get_transitions(*next_pos, next_dir)
            next_num_transitions = fast_count_nonzero(next_transitions)
            opp_dir_options = 1
            for nd in range(4):
                if nd != next_dir:
                    ntrans = raw_env.rail.get_transitions(*next_pos, nd)
                    opp_dir_options = max(opp_dir_options, fast_count_nonzero(ntrans))
            if next_num_transitions == 1:
                if opp_dir_options > 1:
                    # Next cell has choices (merge point ahead)
                    return 'MERGING'
                else:
                    # Next cell is also forward-only
                    return 'FORWARD_ONLY'
            elif next_num_transitions > 1:
                # Next cell is a switch/merge area with alternatives.
                return 'MERGING'
        except Exception:
            pass
        
        # Default fallback
        return 'FORWARD_ONLY'

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
        # By default, trust solver epsilon exactly. A decision-point floor is
        # only applied when explicitly enabled.
        eps_floor = float(getattr(self, 'decision_eps_floor', 0.0))
        if bool(getattr(self, 'use_decision_eps_floor', False)):
            eps_val = max(eps_val, eps_floor)
        if eps_val > 0.0 and np.random.rand() < eps_val:
            # Prefer movement actions during random exploration to avoid
            # collapsing into DO_NOTHING/STOP-heavy local minima.
            move_actions = legal_actions[
                (legal_actions != RailEnvActions.DO_NOTHING)
                & (legal_actions != RailEnvActions.STOP_MOVING)
            ]
            if move_actions.size > 0:
                return int(np.random.choice(move_actions))
            return int(np.random.choice(legal_actions))
        try:
            with torch.no_grad():
                emb = self.encoder_actor.forward_agent(state, handle)
                logits = self.actor_critic_model.actor(emb.unsqueeze(0)).squeeze(0)
                mask_t = torch.from_numpy(mask).to(logits.device)
                # Standard masking (Huang & Ontañón 2022): set illegal logits
                # to a large negative number BEFORE softmax.
                logits = logits.masked_fill(mask_t < 0.5, -1e9)
                # At decision cells, damp idle actions if at least one movement
                # action is legal. This keeps DO_NOTHING/STOP available but
                # reduces their over-selection in sparse-switch layouts.
                cell_type = self._classify_cell_type(agent, self._env.raw_env)
                if cell_type in ('OUTSIDE', 'MERGING', 'SWITCH'):
                    has_move = bool(
                        mask[RailEnvActions.MOVE_LEFT] > 0.5
                        or mask[RailEnvActions.MOVE_FORWARD] > 0.5
                        or mask[RailEnvActions.MOVE_RIGHT] > 0.5
                    )
                    if has_move:
                        idle_pen = float(getattr(self, 'idle_logit_penalty', 1.25))
                        stop_pen = float(getattr(self, 'stop_logit_penalty', 0.80))
                        logits[RailEnvActions.DO_NOTHING] -= idle_pen
                        logits[RailEnvActions.STOP_MOVING] -= stop_pen
                action = Categorical(logits=logits).sample().item()
            return int(action)
        except Exception:
            return int(np.random.choice(legal_actions))

    def act(self, handle: int, state, eps=0.):
        agent: EnvAgent = self._env.raw_env.agents[handle]
        
        # ================================================================
        # Cell-Type-Based Action Selection (State Machine Reduction)
        # Laurent et al. (2021): Optimize credit assignment by only
        # applying policy to genuine decision points (SWITCH/MERGING).
        # ================================================================
        cell_type = self._classify_cell_type(agent, self._env.raw_env)
        
        # FORWARD_ONLY cells: hard-coded MOVE_FORWARD (no policy choice)
        # This avoids training noise on trivial forward-only rail segments.
        if cell_type == 'FORWARD_ONLY':
            return RailEnvActions.MOVE_FORWARD
        
        # DONE: episode complete, no action needed
        if cell_type == 'DONE':
            return RailEnvActions.DO_NOTHING
        
        # OUTSIDE / MERGING / SWITCH: apply policy
        # These are the only meaningful decision points where the RL policy
        # should contribute to credit assignment and learning.
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


def create_decider_agent(observation_space: int, action_space: int, eps: float = 0.0) -> LearningPolicy:
    """Hierarchical Decider policy with Specialist sub-modules + PPO + 1-step
    deadlock BCE aux-loss. See HIERARCHICAL_DECIDER_ARCHITECTURE.md."""
    print('>> DeciderPPOPolicy (Hierarchical Specialists + Decider)')
    print('   - observation_space:', observation_space)
    print('   - action_space:', action_space)
    print('   - temporal_window:', TEMPORAL_WINDOW)
    print(f'   - EPS (epsilon floor): {eps:.4f}')
    policy = DeciderPPOPolicy(
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
    policy.eps_smoothing = eps  # Set epsilon floor
    return policy


ppo_param = MARL_ATTENTION_TEMPORAL_MAPPO_Param(
    hidden_size=64,         # ⬇️ Reduced for 4x faster LSTM (was 128)
    batch_size=256,
    learning_rate=1.4e-4,
    discount=0.99,  # Längere Belohnungsketten
    gae_lambda=0.95,  # Lower variance for stabler PPO updates
    use_gpu=True,
    max_episodes_in_training_memory=12,   # Fresher data -> faster adaptation
    k_epochs=3,
    batch_fraction=0.8,
    max_batches_per_training=10,
    temporal_window=TEMPORAL_WINDOW, # ⚡ MUST MATCH create_temporal_obs_builder_object()!
    encoder_type='lstm'
)

def create_ma_ppo_agent(observation_space: int, action_space: int, eps: float = 0.0, optimizer_mode: str = 'single') -> LearningPolicy:
    """
    Creates PPO Policy with Temporal Transformer Encoder
    
    observation_space: 33 (30 base + 3 velocity)
    eps: Epsilon floor (0.0-1.0)
    optimizer_mode: 'single' = consolidated optimizer, 'multiple' = 4 optimizers with sync decay
    """
    print('>> MARL_ATTENTION_TEMPORAL_PPOPolicy (Temporal Transformer)')
    print('   - observation_space:', observation_space)
    print('   - action_space:', action_space)
    print('   - temporal_window:', TEMPORAL_WINDOW)
    print('   - architecture: 2-Level Attention (Temporal + Spatial)')
    print(f'   - EPS (epsilon floor): {eps:.4f}')
    print(f'   - optimizer_mode: {optimizer_mode}')
        
    policy = MARL_ATTENTION_TEMPORAL_PPOPolicy(
        observation_space,
        action_space,
        ppo_param,
        show_pre_train_debug_msg=False,
        show_progress_bar=True,
        train_frequency=10,    # ⬆️ Train every 10 episodes (faster feedback)
        optimizer_mode=optimizer_mode
    )
    # Avoid late-stage over-conservative clipping that caused the 0.54-0.57 plateau.
    policy.surrogate_eps_clip = 0.15
    policy.weight_entropy = 0.012
    policy.stability_guard_start_episode = 2600
    policy.stability_guard_hard_episode = 3800
    policy.eps_smoothing = eps  # Set epsilon floor
    return policy

def create_ma_ppo_agent_dp(observation_space: int, action_space: int, eps: float = 0.0, optimizer_mode: str = 'single') -> LearningPolicy:
    """
    Creates  PPO Policy with Temporal Transformer Encoder
    
    observation_space: 33 (30 base + 3 velocity)
    eps: Epsilon floor (0.0-1.0)
    optimizer_mode: 'single' = consolidated optimizer, 'multiple' = 4 optimizers with sync decay
    """
    print('>> MARL_ATT_DecisionPointPolicy (Temporal Transformer)')
    print('   - observation_space:', observation_space)
    print('   - action_space:', action_space)
    print('   - temporal_window:', TEMPORAL_WINDOW)
    print('   - architecture: 2-Level Attention (Temporal + Spatial)')
    print(f'   - EPS (epsilon floor): {eps:.4f}')
    print(f'   - optimizer_mode: {optimizer_mode}')
        
    policy = MARL_ATT_DecisionPointPolicy(
        observation_space,
        action_space,
        ppo_param,
        show_pre_train_debug_msg=False,
        show_progress_bar=True,
        train_frequency=10,
        use_deadlock_avoidance_policy=False,
        optimizer_mode=optimizer_mode
    )
    # Stabilized settings for early/mid training: lower KL spikes + stronger exploration.
    policy.surrogate_eps_clip = 0.10
    policy.weight_entropy = 0.045
    policy.reward_scale = 0.06
    policy.weight_loss = 1.4
    policy.stability_guard_start_episode = 1200
    policy.stability_guard_hard_episode = 2600
    policy.ppo_target_kl = 0.025
    policy.ppo_max_kl = 0.050
    policy.ppo_emergency_kl = 0.16
    policy.ppo_emergency_kl_hard = 0.24
    policy.ratio_guard_soft = 1.10
    policy.ratio_guard_soft_low = 0.90
    policy.ratio_guard_hard = 1.15
    policy.ratio_guard_hard_low = 0.85
    policy.max_hard_batches_before_lr_decay = 4
    policy.hard_spike_streak_limit = 3
    policy.actor_lr_min_factor = 0.50
    policy.actor_lr_decay_on_instability = 0.88
    policy.max_eps_random = 0.12
    policy.decision_eps_floor = 0.04
    policy.use_decision_eps_floor = True
    # Sparse-switch maps: keep forward dominant and avoid forcing turn frequency.
    policy.weight_action_diversity = 0.30
    policy.forward_prob_soft_max = 0.58
    policy.lr_prob_soft_min = 0.08
    policy.idle_prob_soft_max = 0.26
    policy.idle_logit_penalty = 3.50
    policy.stop_logit_penalty = 3.20
    policy.eps_smoothing = eps  # Set epsilon floor
    return policy

 
def create_ma_ppo_agent_dp_DLA(observation_space: int, action_space: int, eps: float = 0.0, optimizer_mode: str = 'single') -> LearningPolicy:
    """
    Creates PPO Policy with Temporal Transformer Encoder
    
    observation_space: 33 (30 base + 3 velocity)
    eps: Epsilon floor (0.0-1.0)
    optimizer_mode: 'single' = consolidated optimizer, 'multiple' = 4 optimizers with sync decay
    """
    print('>> MARL_ATT_DecisionPointPolicy with Deadlockavoidance (Temporal Transformer)')
    print('   - observation_space:', observation_space)
    print('   - action_space:', action_space)
    print('   - temporal_window:', TEMPORAL_WINDOW)
    print('   - architecture: 2-Level Attention (Temporal + Spatial)')
    print(f'   - EPS (epsilon floor): {eps:.4f}')
    print(f'   - optimizer_mode: {optimizer_mode}')
    
    policy = MARL_ATT_DecisionPointPolicy(
        observation_space,
        action_space,
        ppo_param,
        show_pre_train_debug_msg=False,
        show_progress_bar=True,
        train_frequency=10,
        use_deadlock_avoidance_policy=True,
        optimizer_mode=optimizer_mode
    )
    policy.surrogate_eps_clip = 0.10
    policy.weight_entropy = 0.045
    policy.reward_scale = 0.06
    policy.weight_loss = 1.4
    policy.stability_guard_start_episode = 2600
    policy.stability_guard_hard_episode = 3800
    # Same anti-stall settings for shielded training.
    policy.ppo_target_kl = 0.025
    policy.ppo_max_kl = 0.050
    policy.ppo_emergency_kl = 0.16
    policy.ppo_emergency_kl_hard = 0.24
    policy.ratio_guard_soft = 1.10
    policy.ratio_guard_soft_low = 0.90
    policy.ratio_guard_hard = 1.15
    policy.ratio_guard_hard_low = 0.85
    policy.max_hard_batches_before_lr_decay = 4
    policy.hard_spike_streak_limit = 3
    policy.actor_lr_min_factor = 0.50
    policy.actor_lr_decay_on_instability = 0.88
    policy.max_eps_random = 0.12
    policy.decision_eps_floor = 0.04
    policy.use_decision_eps_floor = True
    # Sparse-switch maps: keep forward dominant and avoid forcing turn frequency.
    policy.weight_action_diversity = 0.30
    policy.forward_prob_soft_max = 0.58
    policy.lr_prob_soft_min = 0.08
    policy.idle_prob_soft_max = 0.26
    policy.idle_logit_penalty = 3.50
    policy.stop_logit_penalty = 3.20
    policy.eps_smoothing = eps  # Set epsilon floor
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
    """Minimal reward shaper: time cost + bonuses + deadlock penalty.
    
    Key improvement: DO_NOTHING is strongly penalized when action_required=True,
    but NOT penalized when the agent must forward (no decision point).
    This addresses sparse-environment value learning: Value estimates only on
    real decision points, not on trivial forward-only segments.
    """

    STEP_PENALTY = -0.01
    INDIVIDUAL_DONE_BONUS = 10.0
    ALL_DONE_BONUS = 100.0
    DEADLOCK_PENALTY = -8.0
    TIMEOUT_PENALTY = -3.0
    PROGRESS_BONUS = 0.01
    IDLE_STOP_PENALTY = -0.01
    DO_NOTHING_PENALTY = -1.0  # Strong penalty for inaction at decision points
    DO_NOTHING_NOT_ON_MAP_PENALTY = -0.50  # Mild penalty for action attempt before agent on map

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

    def __call__(self, reward, terminal, info, env, actions=None):
        if actions is None:
            actions = {}
        raw_env = env.raw_env
        agents = raw_env.agents
        num_agents = len(agents)
        env_id = id(raw_env)
        episode_done = bool(terminal.get('__all__', False)) if isinstance(terminal, dict) else False

        # Get current distances
        dm = raw_env.distance_map.get()
        distances = np.array([float(dm[a.handle, a.position[0], a.position[1], a.direction]) 
                              if a.position else float('inf') for a in agents], dtype=np.float32)

        # Initialize state if needed (also reinitialize if num_agents changed)
        if env_id not in self._done_charged or len(self._done_charged[env_id]) != num_agents:
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
            deadlocked_now = agent.position is not None and self._is_local_deadlock(raw_env, agent)
            if not self._deadlock_charged[env_id][i] and deadlocked_now:
                self._deadlock_charged[env_id][i] = True
                s += self.DEADLOCK_PENALTY
            
            # Decision-Point Detection: Only penalize DO_NOTHING at genuine switches/merges
            # (NOT on trivial forward-only cells where Value learning is meaningless)
            is_at_decision_point = False
            if agent.position is not None and agent.state.is_on_map_state():
                # Check if agent is at or near a switch (same logic as MARL_ATT_DecisionPointPolicy.act())
                transitions = raw_env.rail.get_transitions(*agent.position, agent.direction)
                num_transitions = fast_count_nonzero(transitions)
                # At a switch: >1 transition option
                # Or near a switch: next cell has multiple transitions (check ahead)
                is_at_decision_point = (num_transitions > 1)
                
                if not is_at_decision_point:
                    # Check if next cell (one forward) is a switch (near-switch detection)
                    try:
                        next_pos = get_new_position(agent.position, agent.direction)
                        next_transitions = raw_env.rail.get_transitions(*next_pos, agent.direction)
                        is_at_decision_point = (fast_count_nonzero(next_transitions) > 1)
                    except:
                        pass
            
            # DO_NOTHING detection and penalty (Sparse-environment fix)
            action_taken = actions.get(agent.handle, RailEnvActions.DO_NOTHING)
            is_do_nothing = (action_taken == RailEnvActions.DO_NOTHING)
            action_required = info.get('action_required', {}).get(agent.handle, True) if isinstance(info, dict) else True
            
            if is_at_decision_point and is_do_nothing and action_required:
                # STRONG penalty: Agent had to decide at switch but chose DO_NOTHING
                s += self.DO_NOTHING_PENALTY
            elif agent.position is None and not is_do_nothing:
                # Mild penalty: Agent tried to move before being on the map
                s += self.DO_NOTHING_NOT_ON_MAP_PENALTY

            # Track progress
            if agent.state.is_on_map_state() and np.isfinite(self._prev_dist[env_id][i]) and np.isfinite(distances[i]):
                self._diag[env_id]['steps'] += 1
                d = distances[i] - self._prev_dist[env_id][i]
                if d < -1e-6:
                    self._diag[env_id]['progress'] += 1
                    # Potential-based shaping: reward if distance to target decreases.
                    s += self.PROGRESS_BONUS
                elif d > 1e-6:
                    self._diag[env_id]['regress'] += 1
                else:
                    self._diag[env_id]['flat'] += 1
                    # Penalize likely unnecessary stop/idle only when action is required
                    # and the agent is not in a local deadlock.
                    if action_required and not deadlocked_now:
                        s += self.IDLE_STOP_PENALTY

            shaped[i] = float(reward[i] + s)

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


class SimpleDoneRewardShaper:
    """Ultra-simple reward shaper: optimize ONLY for done rate and step efficiency.
    
    Reward structure:
    - Every step: -1 (penalize time)
    - Agent reaches done: +100 (one-time bonus, immediate)
    - TIMING FIX: Team bonus is NOT retroactive (agents done at different times).
      Instead, track when all agents are done and reward happens naturally via
      policy.end_episode() when the simulation ends.
    """
    
    STEP_PENALTY = -1.0
    INDIVIDUAL_DONE_BONUS = 100.0
    
    def __init__(self):
        self._done_charged: Dict[int, np.ndarray] = {}  # env_id -> bool array
        self.episode_count = 0
    
    def __call__(self, reward, terminal, info, env, actions=None):
        if actions is None:
            actions = {}
        raw_env = env.raw_env
        agents = raw_env.agents
        num_agents = len(agents)
        env_id = id(raw_env)
        episode_done = bool(terminal.get('__all__', False)) if isinstance(terminal, dict) else False

        # Initialize state if needed
        if env_id not in self._done_charged or len(self._done_charged[env_id]) != num_agents:
            self._done_charged[env_id] = np.zeros(num_agents, dtype=bool)

        # Shaping per agent: -1 per step + 100 when done (one-time)
        shaped = list(reward)
        for i, agent in enumerate(agents):
            s = self.STEP_PENALTY  # -1 per step
            
            # One-time bonus when agent reaches done
            # CRITICAL: This is applied only on the EXACT step agent.state changes to DONE
            if agent.state == TrainState.DONE and not self._done_charged[env_id][i]:
                self._done_charged[env_id][i] = True
                s += self.INDIVIDUAL_DONE_BONUS
                # reward is stored in buffer with this bonus immediately
            
            shaped[i] = float(reward[i] + s)

        if episode_done:
            self.episode_count += 1
            # Clean up for next episode
            if env_id in self._done_charged:
                del self._done_charged[env_id]

        return shaped


# Mode selector: switch between complex and simple reward shapers
_REWARD_SHAPER_MODE = "simple"  # "complex" or "simple"

def set_reward_shaper_mode(mode: str):
    """Set reward shaper mode: 'complex' (PBRS) or 'simple' (done-only)"""
    global _REWARD_SHAPER_MODE
    if mode not in ["complex", "simple"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'complex' or 'simple'")
    _REWARD_SHAPER_MODE = mode
    print(f">> Reward Shaper Mode: {mode.upper()}")

def get_reward_shaper():
    """Get active reward shaper based on mode"""
    if _REWARD_SHAPER_MODE == "complex":
        return FlatlandPBRSShaper()
    elif _REWARD_SHAPER_MODE == "simple":
        return SimpleDoneRewardShaper()
    else:
        raise ValueError(f"Unknown mode: {_REWARD_SHAPER_MODE}")


# Default: use simple shaper for now
flatland_reward_shaper = SimpleDoneRewardShaper()


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
    # Advanced argument parsing with --eps for epsilon floor.
    parser = argparse.ArgumentParser(
        description='MARL Attention Temporal PPO Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python marl_attention_temporal.py final_continue
    python marl_attention_temporal.py final_continue --eps 0.05
    python marl_attention_temporal.py continue --eps 0.1
    python marl_attention_temporal.py continue --eps 0.1 --min_eps 0.001
  python marl_attention_temporal.py eval
        """
    )
    parser.add_argument(
        'mode',
        nargs='?',
        default='',
        choices=['new', 'final', 'final_continue', 'continue', 'eval', 'dla_eval', 'dla'],
        help='Training mode (default: new)'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=1.0,
        metavar='EPS_VALUE',
        dest='eps',
        help='Exploration floor for epsilon-greedy in [0.0, 1.0] (default: 1.0)'
    )

    parser.add_argument(
        '--min_eps',
        type=float,
        default=0.001,
        metavar='MIN_EPS_VALUE',
        dest='min_eps',
        help='Minimum exploration floor for epsilon-greedy in [0.0, 1.0] (default: 0.001)'
    )

    parser.add_argument(
        '--optimizer_mode',
        type=str,
        default='single',
        choices=['single', 'multiple'],
        metavar='MODE',
        dest='optimizer_mode',
        help='Optimizer mode: single = consolidated single optimizer (default), multiple = 4 optimizers with synchronized decay'
    )
    
    args = parser.parse_args()
    mode = args.mode.lower()
    eps = args.eps
    min_eps = args.min_eps
    optimizer_mode = args.optimizer_mode.upper()
    
    # Validate EPS range
    if not (0.0 <= eps <= 1.0):
        print(f"ERROR: --eps must be between 0.0 and 1.0, got {eps}")
        sys.exit(1)
    if not (0.0 <= min_eps <= 1.0):
        print(f"ERROR: --min_eps must be between 0.0 and 1.0, got {min_eps}")
        sys.exit(1)
    min_eps = min(eps, min_eps)  # Use the lower of the two for safety

    print(f"\n[Config] mode={mode}, eps={eps:.4f}, optimizer_mode={optimizer_mode}")
    
    do_rendering = False
    checkpoint_interval = 100  # Default: every 100 episodes
    start_from_phase = 0
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
    elif mode == 'new':
        do_training = True
        test_with_deadlock_avoidance_policy = False
        start_from_phase = 0  # Full curriculum
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
            solver_deadlock.perform_training(max_episodes=5000, min_eps=eps)
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
                environment.get_action_space(),
                eps=eps,
                optimizer_mode=optimizer_mode
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
                        solver.perform_training(
                            max_episodes=phase_episodes,
                            checkpoint_interval=checkpoint_interval,
                            eps=eps,
                            min_eps=min_eps, 
                        )
                else:
                    solver.perform_training(
                        max_episodes=10000,
                        checkpoint_interval=checkpoint_interval,
                        eps=eps,
                        min_eps=min_eps,  # If an epsilon floor is set, use it as min_eps to maintain exploration
                    )
            else:
                solver.load_policy()   
                solver.perform_evaluation(max_episodes=1000)
