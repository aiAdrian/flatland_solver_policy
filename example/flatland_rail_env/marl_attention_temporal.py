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
# USAGE INSTRUCTIONS
# =============================================================================
# This script supports the following modes of operation:
#
# 1. Start New Training:
#    Use this mode to start training from scratch without loading any checkpoints.
#    Command:
#        python marl_attention_temporal.py --train --fresh-start
#
# 2. Continue Training from Checkpoint:
#    Use this mode to resume training from the last saved checkpoint.
#    Command:
#        python marl_attention_temporal.py --train --continue
#    Note:
#        Ensure that the checkpoint exists in the directory:
#        training_output/last_checkpoint/
#
# 3. Evaluation:
#    Use this mode to evaluate the model using the last saved checkpoint.
#    Command:
#        python marl_attention_temporal.py --eval
#    Note:
#        If no checkpoint is found, the evaluation will use the default policy.
#
# 4. Debug Mode:
#    Enable debug mode to get detailed logs during training or evaluation.
#    Command:
#        python marl_attention_temporal.py --train --DEBUG
# =============================================================================

from typing import Callable, Optional, List, Union, Dict
import os
import sys
import argparse
import numpy as np
import torch
from torch.distributions import Categorical

# Ensure project-local imports work when launching this script directly from
# example/flatland_rail_env without manually exporting PYTHONPATH.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flatland.envs.rail_env import RailEnvActions
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.step_utils.states import TrainState
from flatland_railway_extension.RailroadSwitchAnalyser import RailroadSwitchAnalyser
from flatland.envs.fast_methods import fast_count_nonzero, fast_argmax
from environment.environment import Environment
from example.flatland_rail_env.flatland_rail_env_persister import RailEnvironmentPersistable
from policy.learning_policy.learning_policy import LearningPolicy
from rendering.flatland.flatland_simple_renderer import FlatlandSimpleRenderer
from solver.flatland.flatland_solver import FlatlandSolver
from policy.policy import Policy
from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.deadlock_avoidance_policy import DeadLockAvoidancePolicy
from utils.training_evaluation_pipeline import create_random_policy
from marl_attention_temporal_mappo import MARL_ATTENTION_TEMPORAL_PPOPolicy, MARL_ATTENTION_TEMPORAL_MAPPO_Param
from marl_attention_temporal_observation.temporal_multi_agent_observation import TemporalMultiAgentObservation
from marl_attention_temporal_observation.hierarchical_routes_observation import HierarchicalRoutesObservation
from marl_attention_temporal_observation.decision_point_utils import DecisionPointUtils
from decider_policy import DeciderPPOPolicy

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
                 optimizer_mode: str = 'single',
                 use_deadlock_avoidance_policy: bool = False):
        # Base Policy.__init__ calls get_name(), so DLA-related attributes must
        # exist before the parent constructor runs.
        self.use_deadlock_avoidance_policy = use_deadlock_avoidance_policy
        self.deadlock_avoidance_policy: Optional[DeadLockAvoidancePolicy] = None
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
        if self.use_deadlock_avoidance_policy and self.deadlock_avoidance_policy is not None:
            self.deadlock_avoidance_policy.step(handle, state, action, reward, next_state, done)

    def start_step(self, train):
        super(MARL_ATT_DecisionPointPolicy, self).start_step(train)
        if self.use_deadlock_avoidance_policy and self.deadlock_avoidance_policy is not None:
            self.deadlock_avoidance_policy.start_step(train)

    def reset(self, env: Environment):
        self._env = env
        # env.raw_env._max_episode_steps =env.raw_env._max_episode_steps + 100
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
                if cell_type in ('MERGING', 'SWITCH'):
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
        
        # OUTSIDE: deterministic spawn behavior (no policy decision).
        if cell_type == 'OUTSIDE':
            return RailEnvActions.MOVE_FORWARD

        # MERGING / SWITCH: apply policy (only true decision points).
        # These are the only meaningful decision points where the RL policy
        # should contribute to credit assignment and learning.
        action = self._masked_act(handle, state, eps)

        if self.use_deadlock_avoidance_policy and self.deadlock_avoidance_policy is not None:
            if agent.state.is_on_map_state() and action == RailEnvActions.MOVE_FORWARD:
                return self.deadlock_avoidance_policy.act(handle, state, eps)

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
# (90D = 66 base + 24 sparse-neighbor block) as base. The decider policy expects
# this. The legacy 66D DecisionPointObservation works too, but the decider
# performs best with the extended layout.
USE_HIERARCHICAL_OBS = True

def create_temporal_obs_builder_object(debug: bool = False):
    """Factory for TemporalMultiAgentObservation"""
    if USE_HIERARCHICAL_OBS:
        try:
            base = HierarchicalRoutesObservation(debug=debug)
        except TypeError:
            base = HierarchicalRoutesObservation()
        try:
            return TemporalMultiAgentObservation(
                temporal_window=TEMPORAL_WINDOW,
                base_obs=base,
                debug=debug,
            )
        except TypeError:
            return TemporalMultiAgentObservation(
                temporal_window=TEMPORAL_WINDOW,
                base_obs=base,
            )
    try:
        return TemporalMultiAgentObservation(
            temporal_window=TEMPORAL_WINDOW,
            debug=debug,
        )
    except TypeError:
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
        reward_scale=0.12,
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
    learning_rate=2.5e-5,  # Lower LR for high-grad regime; improves PPO update stability
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
    
    observation_space: temporal observation size (66D DecisionPoint or 90D HierarchicalRoutes)
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
    policy.weight_entropy = 0.016  # Erhöhtes Entropie-Gewicht für mehr Exploration
    policy.stability_guard_start_episode = 2600
    policy.stability_guard_hard_episode = 3800
    policy.eps_smoothing = eps  # Set epsilon floor
    return policy


def create_deadlock_avoidance_policy(environment: Environment, action_space: int, show_debug_plot: bool = False) -> Policy:
    return DeadLockAvoidancePolicy(
        environment.get_raw_env(),
        action_space,
        enable_eps=False,
        show_debug_plot=show_debug_plot,
    )

def create_ma_ppo_agent_dp(observation_space: int, action_space: int, eps: float = 0.0, optimizer_mode: str = 'single') -> LearningPolicy:
    """
    Creates  PPO Policy with Temporal Transformer Encoder
    
    observation_space: temporal observation size (66D DecisionPoint or 90D HierarchicalRoutes)
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
        optimizer_mode=optimizer_mode
    )
    # Stabilized settings for early/mid training: lower KL spikes + stronger exploration.
    policy.surrogate_eps_clip = 0.12
    policy.weight_entropy = 0.05
    policy.reward_scale = 0.12  # Increased: done_bonus=30 >> deadlock=-3.6 (ratio 8.3x)
    policy.weight_loss = 1.2
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
    policy.max_eps_random = 0.08
    policy.decision_eps_floor = 0.05
    policy.use_decision_eps_floor = True
    # DONE-only on sparse maps: do not fight naturally high forward usage.
    policy.weight_action_diversity = 0.05
    policy.forward_prob_soft_max = 0.85
    policy.lr_prob_soft_min = 0.08
    policy.idle_prob_soft_max = 0.26
    policy.idle_logit_penalty = 1.40
    policy.stop_logit_penalty = 1.20
    policy.eps_smoothing = eps  # Set epsilon floor
    return policy


def create_ma_ppo_agent_dp_dla(observation_space: int, action_space: int, eps: float = 0.0, optimizer_mode: str = 'single') -> LearningPolicy:
    policy = create_ma_ppo_agent_dp(
        observation_space,
        action_space,
        eps=eps,
        optimizer_mode=optimizer_mode,
    )
    if isinstance(policy, MARL_ATT_DecisionPointPolicy):
        policy.use_deadlock_avoidance_policy = True
    return policy


def create_random_policy_agent(observation_space: int, action_space: int, eps: float = 0.0, optimizer_mode: str = 'single') -> Policy:
    return create_random_policy(observation_space, action_space)


def resolve_policy_creator_list(environment: Environment, policy_mode: Optional[str] = None) -> List[Callable[[int, int], Policy]]:
    if policy_mode is None:
        policy_mode = os.environ.get('FLATLAND_POLICY_MODE', 'pure_marl')
    policy_mode = policy_mode.strip().lower()
    if policy_mode == 'decider':
        return [create_decider_agent]
    if policy_mode in ('pure_marl_dla', 'marl_dla'):
        return [create_ma_ppo_agent_dp_dla]
    if policy_mode in ('dla', 'dead_lock_avoidance', 'deadlock_avoidance'):
        return [lambda observation_space, action_space, eps=0.0, optimizer_mode='single': create_deadlock_avoidance_policy(environment, action_space)]
    if policy_mode == 'random':
        return [create_random_policy_agent]
    return [create_ma_ppo_agent_dp]

if __name__ == "__main__":
    # Advanced argument parsing with --eps for epsilon floor.
    parser = argparse.ArgumentParser(
        description='MARL Attention Temporal PPO Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python marl_attention_temporal.py --train --fresh-start
  python marl_attention_temporal.py --train --continue --eps 0.1
  python marl_attention_temporal.py --train --continue --eps 0.1 --min_eps 0.001
  python marl_attention_temporal.py --eval
  python marl_attention_temporal.py final_continue --eps 0.0
  python marl_attention_temporal.py --train final --eps 0.0
  python marl_attention_temporal.py final --eps 0.0
"""
    )
    parser.add_argument(
        'mode',
        nargs='?',
        default='new',
        choices=['new', 'final', 'final_continue', 'continue', 'eval'],
        help='Training mode (default: new)'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        dest='legacy_train',
        help='Legacy flag: run training mode'
    )
    parser.add_argument(
        '--fresh-start',
        action='store_true',
        dest='legacy_fresh_start',
        help='Legacy flag: train from scratch (maps to mode=new)'
    )
    parser.add_argument(
        '--continue',
        action='store_true',
        dest='legacy_continue',
        help='Legacy flag: continue training from checkpoint (maps to mode=continue)'
    )
    parser.add_argument(
        '--eval',
        action='store_true',
        dest='legacy_eval',
        help='Legacy flag: evaluation mode (maps to mode=eval)'
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
    
    parser.add_argument(
        '--rendering',
        action='store_true',
        dest='rendering',
        help='Enable FlatlandSimpleRenderer during evaluation/training runs'
    )

    parser.add_argument(
        '--policy_mode',
        type=str,
        default='pure_marl',
        choices=['pure_marl', 'pure_marl_dla', 'marl_dla', 'dla', 'dead_lock_avoidance', 'deadlock_avoidance', 'random', 'decider'],
        metavar='POLICY_MODE',
        dest='policy_mode',
        help=(
            "Policy selector:\n"
            "  pure_marl            -> PPO DecisionPoint policy (ohne DLA, Standard)\n"
            "  pure_marl_dla        -> PPO DecisionPoint + DLA-Shield\n"
            "  marl_dla             -> Alias fuer pure_marl_dla\n"
            "  dla                  -> nur DeadLockAvoidancePolicy (heuristisch)\n"
            "  dead_lock_avoidance  -> Alias fuer dla\n"
            "  deadlock_avoidance   -> Alias fuer dla\n"
            "  random               -> RandomPolicy (Baseline)\n"
            "  decider              -> DeciderPPOPolicy"
        )
    )
    
    parser.add_argument(
        '--DEBUG',
        action='store_true',
        dest='debug',
        help='Enable debug mode for DecisionPointObservation (default: off)'
    )

    args = parser.parse_args()
    mode = args.mode.lower()
    if args.legacy_eval:
        mode = 'eval'
    elif args.legacy_continue:
        mode = 'continue'
    elif args.legacy_fresh_start:
        mode = 'new'
    elif args.legacy_train and mode == 'eval':
        mode = 'new'
    eps = args.eps
    min_eps = args.min_eps
    optimizer_mode = args.optimizer_mode.upper()
    rendering = bool(args.rendering)
    policy_mode = args.policy_mode.strip().lower()
    debug_mode = args.debug  # Capture debug flag
    do_training = mode != 'eval'
    do_rendering = rendering
    checkpoint_interval = 50
    start_from_phase = 0

    # Validate EPS range
    if not (0.0 <= eps <= 1.0):
        print(f"ERROR: --eps must be between 0.0 and 1.0, got {eps}")
        sys.exit(1)
    if not (0.0 <= min_eps <= 1.0):
        print(f"ERROR: --min_eps must be between 0.0 and 1.0, got {min_eps}")
        sys.exit(1)
    min_eps = min(eps, min_eps)  # Use the lower of the two for safety

    print(f"\n[Config] mode={mode}, eps={eps:.4f}, optimizer_mode={optimizer_mode}, policy_mode={policy_mode}, debug={debug_mode}")

    environment = RailEnvironmentPersistable(
        obs_builder_object_creator=lambda: create_temporal_obs_builder_object(debug=debug_mode),
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
    policy_creator_list = resolve_policy_creator_list(environment, policy_mode=policy_mode)

    # Use the actual base-obs size so the policy gets a matching state_size
    # (64D for DecisionPointObservation, 88D for HierarchicalRoutesObservation).
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
        # ----------------------------------------------------------------
        import os as _os_il
        _il_ckpt = _os_il.environ.get('IL_LOAD_CHECKPOINT', '').strip()
        if not _il_ckpt:
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
        latest_ckpt = f"training_output/last_checkpoint/{solver.get_name()}_{solver.policy.get_name()}"
        
        if do_training:
            if mode == 'continue' or mode == 'final_continue':
                if mode == 'final_continue':
                    # Load from training_output/last_checkpoint/ for automatic recovery
                    solver.load_policy(filename=latest_ckpt)
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
            # Eval should mirror final_continue loading to avoid stale/shape-mismatched legacy checkpoints.
            actor_ckpt_file = latest_ckpt + ".actor"
            if os.path.exists(actor_ckpt_file):
                solver.load_policy(filename=latest_ckpt)
                print("✅ Loaded eval checkpoint from training_output/last_checkpoint/")
            else:
                print("⚠️ No last_checkpoint found; falling back to default policy path.")
                solver.load_policy()
            solver.perform_evaluation(max_episodes=1000)
