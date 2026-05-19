# =============================================================================
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportIncompatibleMethodOverride=false, reportCallIssue=false, reportAssignmentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false

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

from typing import Callable, Optional, List, Union
import os
import sys
import inspect
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

from flatland.envs.rail_env import RailEnvActions  # noqa: E402
from flatland.core.grid.grid4_utils import get_new_position  # noqa: E402
from flatland.envs.agent_utils import EnvAgent  # noqa: E402
from flatland.envs.step_utils.states import TrainState  # noqa: E402
from flatland_railway_extension.RailroadSwitchAnalyser import RailroadSwitchAnalyser  # noqa: E402
from flatland.envs.fast_methods import fast_count_nonzero, fast_argmax  # noqa: E402
from environment.environment import Environment  # noqa: E402
from example.flatland_rail_env.flatland_rail_env_persister import RailEnvironmentPersistable  # noqa: E402
from policy.learning_policy.learning_policy import LearningPolicy  # noqa: E402
from rendering.flatland.flatland_simple_renderer import FlatlandSimpleRenderer  # noqa: E402
from solver.flatland.flatland_solver import FlatlandSolver  # noqa: E402
from policy.policy import Policy  # noqa: E402
from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.deadlock_avoidance_policy import DeadLockAvoidancePolicy  # noqa: E402
from utils.training_evaluation_pipeline import create_random_policy  # noqa: E402
from marl_attention_temporal_mappo import MARL_ATTENTION_TEMPORAL_PPOPolicy, MARL_ATTENTION_TEMPORAL_MAPPO_Param  # noqa: E402
from marl_attention_temporal_observation.temporal_multi_agent_observation import TemporalMultiAgentObservation  # noqa: E402
from marl_attention_temporal_observation.hierarchical_routes_observation import HierarchicalRoutesObservation  # noqa: E402
from marl_attention_temporal_observation.decision_point_utils import DecisionPointUtils  # noqa: E402
from decider_policy import DeciderPPOPolicy  # noqa: E402

# Runtime/device config.
# CPU-first default: this workload contains many small Python-side operations
# (tree payload assembly/message passing), where GPU can be slower due to
# transfer and launch overhead. Enable GPU explicitly via FLATLAND_USE_GPU=1.
FORCE_CPU = str(os.getenv('FLATLAND_FORCE_CPU', '0')).strip().lower() in ('1', 'true', 'yes', 'on')
USE_GPU_REQUESTED = str(os.getenv('FLATLAND_USE_GPU', '0')).strip().lower() in ('1', 'true', 'yes', 'on')
USE_GPU_EFFECTIVE = (not FORCE_CPU) and USE_GPU_REQUESTED and torch.cuda.is_available()
device = torch.device('cuda' if USE_GPU_EFFECTIVE else 'cpu')


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


REWARD_STEP_PENALTY = _env_float('FLATLAND_REWARD_STEP_PENALTY', 0.01)
REWARD_DONE_BONUS = _env_float('FLATLAND_REWARD_DONE_BONUS', 10.0)  # Boosted for stronger positive signal
REWARD_ALL_DONE_BONUS = _env_float('FLATLAND_REWARD_ALL_DONE_BONUS', 100.0)
REWARD_DEADLOCK_PENALTY = _env_float('FLATLAND_DEADLOCK_PENALTY', 100.0)  # Lowered for less negative bias
REWARD_PROGRESS_BONUS = _env_float('FLATLAND_REWARD_PROGRESS_BONUS', 0.05)  # ↑ reward for progress
FINAL_NOT_SOLVED_PENALTY = _env_float('FLATLAND_FINAL_NOT_SOLVED_PENALTY', 10.0)  # Large penalty for not solving by episode end

class FlatlandSparseRewardShaper:
    """Reward shaping for sparse/deadlock-heavy Flatland training.

    Per step per agent:
    - step penalty: -step_penalty
    - done bonus: +done_bonus (once per agent)
    - all-done bonus: +all_done_bonus (once for each agent when all are done)
    - deadlock penalty: -deadlock_penalty (for on-map non-done deadlocked agents)
    - progress bonus: +progress_bonus when distance-to-target decreases
    """

    def __init__(
        self,
        step_penalty: float,
        done_bonus: float,
        all_done_bonus: float,
        deadlock_penalty: float,
        progress_bonus: float,
        final_not_solved_penalty: float
    ):
        self.step_penalty = float(step_penalty)
        self.done_bonus = float(done_bonus)
        self.all_done_bonus = float(all_done_bonus)
        self.deadlock_penalty = float(deadlock_penalty)
        self.progress_bonus = float(progress_bonus)
        self.final_not_solved_penalty = float(final_not_solved_penalty)
        self._rewarded_done = {}
        self._all_done_bonus_given = False
        self._prev_distance = {}
        self._last_episode_deadlock_count = 0
        self._current_episode_deadlocks = set()

    def _reset_episode_state(self, env: Environment):
        n_agents = int(len(env.raw_env.agents))
        self._rewarded_done = {int(a.handle): False for a in env.raw_env.agents}
        self._all_done_bonus_given = False
        self._prev_distance = {}
        self._last_episode_deadlock_count = int(len(self._current_episode_deadlocks))
        self._current_episode_deadlocks = set()
        if len(self._rewarded_done) != n_agents:
            self._rewarded_done = {idx: False for idx in range(n_agents)}
        for agent in env.raw_env.agents:
            self._prev_distance[int(agent.handle)] = self._current_agent_distance(env, agent)

    def get_last_episode_deadlock_count(self) -> int:
        return int(self._last_episode_deadlock_count)

    @staticmethod
    def _current_agent_distance(env: Environment, agent: EnvAgent) -> float:
        pos = agent.position if agent.position is not None else agent.initial_position
        direction = agent.direction if agent.direction is not None else agent.initial_direction
        if pos is None or direction is None:
            return np.inf
        try:
            dist = float(env.raw_env.distance_map.get()[int(agent.handle), pos[0], pos[1], int(direction)])
        except Exception:
            return np.inf
        return dist if np.isfinite(dist) else np.inf

    @staticmethod
    def _build_agent_map(env: Environment) -> np.ndarray:
        raw_env = env.raw_env
        agent_map = np.zeros((raw_env.height, raw_env.width), dtype=np.int32) - 1
        for agent in raw_env.agents:
            if agent.position is not None:
                agent_map[agent.position] = int(agent.handle)
        return agent_map

    def __call__(self, reward, terminal, info, env: Environment, actions=None):
        raw_env = env.raw_env
        if raw_env._elapsed_steps <= 1:
            self._reset_episode_state(env)

        shaped = dict(reward)
        agent_map = self._build_agent_map(env)
        all_agents_done = all(agent.state == TrainState.DONE for agent in raw_env.agents) 
        active_on_map_handles = [
            int(a.handle)
            for a in raw_env.agents
            if a.state != TrainState.DONE and a.position is not None and a.direction is not None
        ]
        deadlock_check_enabled = len(active_on_map_handles) > 1
        if raw_env._elapsed_steps > (raw_env._max_episode_steps -5):
            all_agents_done = False  # Don't give all-done bonus if episode ended due to step limit.    
 
        for handle in env.get_agent_handles():
            agent = raw_env.agents[handle]
            
            r = 0.0
                
            if agent.state > TrainState.WAITING:
                # Apply time pressure for every non-terminal agent so idling is costly.
                if agent.state < TrainState.DONE:
                    r -= self.step_penalty

                    if agent.position is not None and agent.direction is not None:
                        current_dist = self._current_agent_distance(env, agent)
                        prev_dist = float(self._prev_distance.get(handle, current_dist))
                        if current_dist < prev_dist:
                            r += self.progress_bonus
                        self._prev_distance[handle] = current_dist

                        if deadlock_check_enabled and DecisionPointUtils.is_local_deadlock(raw_env, agent, agent_map):
                            self._current_episode_deadlocks.add(int(handle))
                            r -= self.deadlock_penalty

                # +BONUS once when an agent reaches target.
                if agent.state == TrainState.DONE and not bool(self._rewarded_done.get(handle, False)):
                    r = self.done_bonus
                    self._rewarded_done[handle] = True
 
                # If all agents are done, grant one-time team bonus to each agent.
                if all_agents_done and not self._all_done_bonus_given:
                    r = self.all_done_bonus

                if raw_env._elapsed_steps > (raw_env._max_episode_steps -5):
                    r = -self.final_not_solved_penalty

            shaped[handle] = float(r)

        if all_agents_done and not self._all_done_bonus_given:
            self._all_done_bonus_given = True

        if terminal.get('__all__', False):
            self._last_episode_deadlock_count = int(len(self._current_episode_deadlocks))
 
        return shaped


class MARL_ATT_DecisionPointPolicy(MARL_ATTENTION_TEMPORAL_PPOPolicy):

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 in_parameters: Union[MARL_ATTENTION_TEMPORAL_MAPPO_Param, None] = None, 
                 show_pre_train_debug_msg=False,
                 show_progress_bar=True,
                 train_frequency=10,
                 optimizer_mode: str = 'single',
                 use_deadlock_avoidance_policy: bool = False,
                 use_action_masking: bool = True):
        # Base Policy.__init__ calls get_name(), so DLA-related attributes must
        # exist before the parent constructor runs.
        self.use_deadlock_avoidance_policy = use_deadlock_avoidance_policy
        self.use_action_masking = use_action_masking  # A/B: compare masked vs unmasked action selection
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
        self.force_forward_on_forward_only = str(
            os.getenv('FLATLAND_FORCE_FORWARD_ON_FORWARD_ONLY', '0')
        ).strip().lower() in ('1', 'true', 'yes', 'on')
        self.stop_action_floor = float(np.clip(
            _env_float('FLATLAND_STOP_ACTION_FLOOR', 0.02),
            0.0,
            0.20,
        ))
        # Route prior from DecisionPointObservation base features [12:15]
        # (sp_left/sp_forward/sp_right). This keeps navigation simple and
        # stable while still allowing PPO exploration around merges/switches.
        self.sp_hint_route_prior_prob = float(np.clip(
            _env_float('FLATLAND_SP_HINT_ROUTE_PRIOR_PROB', 0.65),
            0.0,
            1.0,
        ))
        self.sp_hint_logit_bonus = float(np.clip(
            _env_float('FLATLAND_SP_HINT_LOGIT_BONUS', 1.25),
            0.0,
            4.0,
        ))

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

    def end_episode(self, train):
        # Call parent logic
        super().end_episode(train)
        # Avoid per-episode I/O overhead; print every N episodes.
        log_interval = max(1, int(_env_int('FLATLAND_LOG_INTERVAL', 20)))
        self._episode_log_counter = int(getattr(self, '_episode_log_counter', 0)) + 1
        if self._episode_log_counter % log_interval != 0:
            return
        # --- LOGGING: Action distribution and reward per episode ---
        import collections
        action_hist = collections.Counter()
        reward_hist = []
        if hasattr(self, 'current_episode_memory'):
            for handle in range(len(self.current_episode_memory)):
                transitions = self.current_episode_memory.get_transitions(handle)
                for t in transitions:
                    if len(t) > 1:
                        action_hist[t[1]] += 1
                    if len(t) > 2:
                        reward_hist.append(t[2])
        total = sum(action_hist.values())
        if total > 0:
            print("[ActionDist]", {a: f"{c/total:.2%}" for a, c in sorted(action_hist.items())})
        if reward_hist:
            import numpy as np
            print(f"[Reward] mean={np.mean(reward_hist):.3f} min={np.min(reward_hist):.3f} max={np.max(reward_hist):.3f}")

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
        next_pos = get_new_position(agent.position, agent.direction)
        if next_pos[0] < 0 or next_pos[0] >= raw_env.height or next_pos[1] < 0 or next_pos[1] >= raw_env.width:
            return 'FORWARD_ONLY'
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

        if agent.state == TrainState.DONE:
            mask[RailEnvActions.DO_NOTHING] = 1.0
            return mask

        if agent.state == TrainState.WAITING:
            mask[RailEnvActions.DO_NOTHING] = 1.0
            return mask


        # If agent is not yet on the map, only DO_NOTHING / MOVE_FORWARD make 
        if agent.state.is_off_map_state():
            # Outside the rail map, force spawn progress only.
            mask[RailEnvActions.DO_NOTHING] = 1.0
            mask[RailEnvActions.STOP_MOVING] = 1.0
            mask[RailEnvActions.MOVE_FORWARD] = 1.0
            return mask


        mask[RailEnvActions.STOP_MOVING] = 1.0

        position, direction = self._get_agent_position_and_direction(agent)
        if position is None or direction is None:
            mask[RailEnvActions.MOVE_FORWARD] = 1.0
            return mask
 
        transitions = self._env.raw_env.rail.get_transitions(*position, direction)
        # Map (new_direction relative to current direction) -> RailEnvActions.
        # Flatland convention: forward = same direction;
        # left = (direction - 1) % 4; right = (direction + 1) % 4;
        # back = (direction + 2) % 4 (only legal at dead-ends).      
        if fast_count_nonzero(transitions) == 1:
            mask[RailEnvActions.MOVE_FORWARD] = 1.0
        else:
            if transitions[ (direction - 1) % 4]:
                mask[RailEnvActions.MOVE_LEFT] = 1.0
            if transitions[ (direction) % 4]:
                mask[RailEnvActions.MOVE_FORWARD] = 1.0
            if transitions[ (direction + 1) % 4]:
                mask[RailEnvActions.MOVE_RIGHT] = 1.0
 
        return mask

    def _masked_act(self, handle: int, state, eps: float, cell_type: str) -> int:
        """Sample from the actor with invalid-action masking.

        Falls back to the unmasked parent implementation if the encoder
        is not yet available (defensive)."""
        agent: EnvAgent = self._env.raw_env.agents[handle]
        mask = self._legal_action_mask(agent)
        legal_actions = np.flatnonzero(mask > 0.5)
        if legal_actions.size == 0:
            return int(super(MARL_ATT_DecisionPointPolicy, self).act(handle, state, eps))

        sp_hint_action = self._extract_shortest_path_hint_action(state, mask)

        # First-order navigation prior: at decision points, follow shortest-path
        # hint frequently so learning does not waste updates rediscovering trivial routing.
        if sp_hint_action is not None and self.sp_hint_route_prior_prob > 0.0:
            if np.random.rand() < self.sp_hint_route_prior_prob:
                return int(sp_hint_action)

        # Optional minimal STOP exploration floor so STOP does not collapse to 0
        # merely due policy initialization or narrow early trajectories.
        if self.stop_action_floor > 0.0:
            if mask[RailEnvActions.STOP_MOVING] > 0.5 and np.random.rand() < self.stop_action_floor:
                return int(RailEnvActions.STOP_MOVING)

        # Make solver epsilon meaningful: random among legal rail actions.
        eps_val = float(eps) if eps is not None else 0.0
        # By default, trust solver epsilon exactly. A decision-point floor is
        # only applied when explicitly enabled.
        eps_floor = float(getattr(self, 'decision_eps_floor', 0.0))
        if bool(getattr(self, 'use_decision_eps_floor', False)):
            eps_val = max(eps_val, eps_floor)
        if eps_val > 0.0 and np.random.rand() < eps_val:
            # Pure masked exploration: sample uniformly from legal actions.
            return int(np.random.choice(legal_actions))
        with torch.no_grad():
            emb = self.encoder_actor.forward_agent(state, handle)
            logits = self.actor_critic_model.actor(emb.unsqueeze(0)).squeeze(0)
            mask_t = torch.from_numpy(mask).to(logits.device)
            # Standard masking (Huang & Ontañón 2022): set illegal logits
            # to a large negative number BEFORE softmax.
            logits = logits.masked_fill(mask_t < 0.5, -1e9)
            if sp_hint_action is not None and self.sp_hint_logit_bonus > 0.0:
                logits[int(sp_hint_action)] += float(self.sp_hint_logit_bonus)
            # At decision cells, damp idle actions if at least one movement
            # action is legal. This keeps DO_NOTHING/STOP available but
            # reduces their over-selection in sparse-switch layouts.
            if False:
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

    def _extract_shortest_path_hint_action(self, state, mask: np.ndarray) -> Optional[int]:
        """Read DecisionPointObservation shortest-path one-hot from base features [12:15].

        Returns a legal RailEnv action id (LEFT/FORWARD/RIGHT) or None.
        """
        try:
            latest = state[-1] if isinstance(state, (list, tuple)) and len(state) > 0 else state
            obs_vec = latest[0] if isinstance(latest, (list, tuple)) and len(latest) > 0 else latest
            obs = np.asarray(obs_vec, dtype=np.float32).reshape(-1)
        except Exception:
            return None

        if obs.shape[0] < 15:
            return None

        hints = obs[12:15]
        if not np.all(np.isfinite(hints)):
            return None

        idx = int(np.argmax(hints))
        if float(hints[idx]) < 0.5:
            return None

        idx_to_action = {
            0: int(RailEnvActions.MOVE_LEFT),
            1: int(RailEnvActions.MOVE_FORWARD),
            2: int(RailEnvActions.MOVE_RIGHT),
        }
        action = idx_to_action.get(idx)
        if action is None:
            return None
        if action >= len(mask) or mask[action] <= 0.5:
            return None
        return int(action)

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

        # OUTSIDE should always spawn deterministically to avoid no-op learning noise.
        if cell_type == 'OUTSIDE':
            return RailEnvActions.MOVE_FORWARD
        
        # MERGING / SWITCH: apply policy (only true decision points).
        # These are the only meaningful decision points where the RL policy
        # should contribute to credit assignment and learning.
        # A/B comparison: use_action_masking toggles between masked (safe rail actions)
        # and unmasked (full policy freedom). Set FLATLAND_USE_ACTION_MASKING=0 to disable.
        if self.use_action_masking:
            action = self._masked_act(handle, state, eps, cell_type)  # with masking
        else:
            action = super(MARL_ATT_DecisionPointPolicy, self).act(handle, state, eps)  # without masking

        if self.use_deadlock_avoidance_policy and self.deadlock_avoidance_policy is not None:
            if agent.state.is_on_map_state() and action == RailEnvActions.MOVE_FORWARD:
                return self.deadlock_avoidance_policy.act(handle, state, eps)

        return action
 

# =============================================================================
# ENVIRONMENT & TRAINING SETUP
# =============================================================================

# Globale Variable für die temporale Fenstergröße
TEMPORAL_WINDOW = 3  # 3 Frames -> Bewegung/Velocity wird durch Temporal-Attention nutzbar
# Local tree-search horizon for DecisionPointObservation.
# 6 is a strong default on dense merge topologies; 5 is faster but may miss
# deeper backward-inflow conflicts.
LOCAL_TREE_SEARCH_DEPTH = 12
LOCAL_TREE_RANDOM_START_DEPTH = 2
LOCAL_TREE_MAX_SIDE_BRANCHES = 1
LOCAL_TREE_DISTANCE_BIAS = 2.0
LOCAL_TREE_MODE = 'stochastic'
LOCAL_TREE_MCTS_ROLLOUTS = 6
LOCAL_TREE_MCTS_HORIZON = 4
LOCAL_TREE_UCB_C = 1.2
LOCAL_TREE_CONTRACT_DEPTH = 8
LOCAL_TREE_MAX_NODES = 72
LOCAL_TREE_MIN_NODES = 32
LOCAL_TREE_ADAPTIVE_BUDGET = 'on'
LOCAL_TREE_ADAPTIVE_BRANCH_BONUS = 6
LOCAL_TREE_ADAPTIVE_CONFLICT_BONUS = 14
LOCAL_TREE_ADAPTIVE_DEPTH_BONUS = 2
LOCAL_TREE_DEADLOCK_PROBE_DEPTH = 7
LOCAL_TREE_DEADLOCK_MAX_STATES = 96
LOCAL_TREE_CLIP_FEATURES = 'on'

# ========================================================================
# DYNAMIC AGENT COUNT CONFIG: Override via FLATLAND_SIMPLIFIED_MAPPO env var
# ========================================================================
# FLATLAND_SIMPLIFIED_MAPPO=0           → Full Mode (standard 5-agent config)
# FLATLAND_SIMPLIFIED_MAPPO=N           → Core PPO with N agents
# FLATLAND_SIMPLIFIED_MAPPO=1,5,10,100  → Core PPO sweep in one run
# ========================================================================
simplified_agent_count = os.getenv('FLATLAND_SIMPLIFIED_MAPPO', '0').strip()
try:
    _simplified_n = int(simplified_agent_count)
    if _simplified_n > 0:
        PURE_MARL_AGENT_COUNTS = [_simplified_n]
        print(f"[Config] CORE PPO MODE: {_simplified_n} agent(s) (from FLATLAND_SIMPLIFIED_MAPPO)")
    else:
        PURE_MARL_AGENT_COUNTS = [5]  # Default full mode
except ValueError:
    parts = [p.strip() for p in simplified_agent_count.split(',') if p.strip()]
    parsed = []
    for p in parts:
        try:
            n = int(p)
            if n > 0:
                parsed.append(n)
        except ValueError:
            continue
    if parsed:
        PURE_MARL_AGENT_COUNTS = sorted(set(parsed))
        print(f"[Config] CORE PPO SWEEP MODE: agents={PURE_MARL_AGENT_COUNTS} (from FLATLAND_SIMPLIFIED_MAPPO)")
    else:
        PURE_MARL_AGENT_COUNTS = [5]  # Default full mode

# High-success curriculum: bias training toward hard coordination cases
# while keeping a small share of easy cases for stability.
PURE_MARL_MAX_AGENTS = max(PURE_MARL_AGENT_COUNTS)

# Auto speed profile for local-search complexity.
# Goal: keep training throughput stable as agent count scales (5/10/20/100)
# without requiring manual CLI/env tweaks.
auto_speed_profile = str(os.getenv('FLATLAND_AUTO_SPEED_PROFILE', '1')).strip().lower() in ('1', 'true', 'yes', 'on')
if auto_speed_profile:
    if PURE_MARL_MAX_AGENTS >= 20:
        LOCAL_TREE_SEARCH_DEPTH = 8
        LOCAL_TREE_CONTRACT_DEPTH = 6
        LOCAL_TREE_MAX_NODES = 48
        LOCAL_TREE_MIN_NODES = 20
        LOCAL_TREE_DEADLOCK_PROBE_DEPTH = 5
        LOCAL_TREE_DEADLOCK_MAX_STATES = 64
    elif PURE_MARL_MAX_AGENTS >= 10:
        LOCAL_TREE_SEARCH_DEPTH = 9
        LOCAL_TREE_CONTRACT_DEPTH = 7
        LOCAL_TREE_MAX_NODES = 56
        LOCAL_TREE_MIN_NODES = 24
        LOCAL_TREE_DEADLOCK_PROBE_DEPTH = 6
        LOCAL_TREE_DEADLOCK_MAX_STATES = 80
    elif PURE_MARL_MAX_AGENTS >= 5:
        LOCAL_TREE_SEARCH_DEPTH = 10
        LOCAL_TREE_CONTRACT_DEPTH = 7
        LOCAL_TREE_MAX_NODES = 64
        LOCAL_TREE_MIN_NODES = 28
        LOCAL_TREE_DEADLOCK_PROBE_DEPTH = 6
        LOCAL_TREE_DEADLOCK_MAX_STATES = 80
    print(
        f"[Config] auto_speed_profile=on: depth={LOCAL_TREE_SEARCH_DEPTH}, "
        f"contract_depth={LOCAL_TREE_CONTRACT_DEPTH}, max_nodes={LOCAL_TREE_MAX_NODES}, "
        f"min_nodes={LOCAL_TREE_MIN_NODES}, probe_depth={LOCAL_TREE_DEADLOCK_PROBE_DEPTH}, "
        f"max_states={LOCAL_TREE_DEADLOCK_MAX_STATES}"
    )

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
    {'name': 'phase0_nav',   'agent_counts': [1],          'num_envs': 10, 'episodes': 500},
    {'name': 'phase1_solo',  'agent_counts': [1, 2],        'num_envs': 12, 'episodes': 500},
    {'name': 'phase2_easy',  'agent_counts': [2, 3, 4],     'num_envs': 16, 'episodes': 500},
    {'name': 'phase3_mid',   'agent_counts': [3, 4, 5],     'num_envs': 18, 'episodes': 500},
    {'name': 'phase4_hard',  'agent_counts': [4, 5],        'num_envs': 22, 'episodes': 500},
    {'name': 'phase5_final', 'agent_counts': [5], 'num_envs': 50, 'episodes': 8000},
]

if INCLUDE_5_AGENTS_IN_FINAL:
    CURRICULUM_PHASES[-1]['agent_counts'] = [1, 2, 3, 4, 5]
    CURRICULUM_PHASES[-1]['episodes'] = 10000

# Toggle: when True, the temporal wrapper uses HierarchicalRoutesObservation
# (48D = 24 base + 24 sparse-neighbor block) as base. The decider policy expects
# this. The legacy larger DecisionPointObservation versions are no longer active;
# current base contract is 24D.
# performs best with the extended layout.
USE_HIERARCHICAL_OBS = False

def create_temporal_obs_builder_object(
    debug: bool = False,
    search_depth: int = LOCAL_TREE_SEARCH_DEPTH,
    random_start_depth: int = LOCAL_TREE_RANDOM_START_DEPTH,
    max_side_branches: int = LOCAL_TREE_MAX_SIDE_BRANCHES,
    distance_bias: float = LOCAL_TREE_DISTANCE_BIAS,
    tree_mode: str = LOCAL_TREE_MODE,
    mcts_rollouts: int = LOCAL_TREE_MCTS_ROLLOUTS,
    mcts_horizon: int = LOCAL_TREE_MCTS_HORIZON,
    ucb_c: float = LOCAL_TREE_UCB_C,
    contract_depth: int = LOCAL_TREE_CONTRACT_DEPTH,
    max_nodes: int = LOCAL_TREE_MAX_NODES,
    min_nodes: int = LOCAL_TREE_MIN_NODES,
    adaptive_budget: str = LOCAL_TREE_ADAPTIVE_BUDGET,
    adaptive_branch_bonus: int = LOCAL_TREE_ADAPTIVE_BRANCH_BONUS,
    adaptive_conflict_bonus: int = LOCAL_TREE_ADAPTIVE_CONFLICT_BONUS,
    adaptive_depth_bonus: int = LOCAL_TREE_ADAPTIVE_DEPTH_BONUS,
    deadlock_probe_depth: int = LOCAL_TREE_DEADLOCK_PROBE_DEPTH,
    deadlock_max_states: int = LOCAL_TREE_DEADLOCK_MAX_STATES,
    clip_tree_features: str = LOCAL_TREE_CLIP_FEATURES,
):
    """Build TemporalMultiAgentObservation with strict 24D-base + raw-tree contract.

    Contract used by the MAPPO encoder pipeline:
    - Base observation vector: fixed 24D from DecisionPointObservation.
    - Tree context: provided only via raw payload (nodes/edges/seen_agents),
      not serialized into the base vector.
    - Opponents: supplied through seen_agents-based temporal wrapper selection.

    The function applies CLI tree-search parameters directly to the underlying
    base observation builder and returns a temporal wrapper with window size
    TEMPORAL_WINDOW.
    """

    def _apply_tree_search_cfg(base_obs):
        if hasattr(base_obs, 'search_depth'):
            base_obs.search_depth = max(1, int(search_depth))
        if hasattr(base_obs, 'local_search_random_start_depth'):
            base_obs.local_search_random_start_depth = max(0, int(random_start_depth))
        if hasattr(base_obs, 'local_search_max_side_branches'):
            base_obs.local_search_max_side_branches = max(0, int(max_side_branches))
        if hasattr(base_obs, 'local_search_distance_bias'):
            base_obs.local_search_distance_bias = max(0.1, float(distance_bias))
        if hasattr(base_obs, 'local_search_mode'):
            base_obs.local_search_mode = str(tree_mode).lower()
        if hasattr(base_obs, 'local_search_mcts_rollouts'):
            base_obs.local_search_mcts_rollouts = max(1, int(mcts_rollouts))
        if hasattr(base_obs, 'local_search_mcts_horizon'):
            base_obs.local_search_mcts_horizon = max(1, int(mcts_horizon))
        if hasattr(base_obs, 'local_search_ucb_c'):
            base_obs.local_search_ucb_c = max(0.01, float(ucb_c))
        if hasattr(base_obs, 'local_search_contract_depth'):
            base_obs.local_search_contract_depth = max(0, int(contract_depth))
        if hasattr(base_obs, 'local_search_max_nodes'):
            base_obs.local_search_max_nodes = max(8, int(max_nodes))
        if hasattr(base_obs, 'local_search_min_nodes'):
            base_obs.local_search_min_nodes = max(8, int(min_nodes))
        if hasattr(base_obs, 'local_search_adaptive_budget'):
            base_obs.local_search_adaptive_budget = str(adaptive_budget).lower() == 'on'
        if hasattr(base_obs, 'local_search_adaptive_branch_bonus'):
            base_obs.local_search_adaptive_branch_bonus = max(0, int(adaptive_branch_bonus))
        if hasattr(base_obs, 'local_search_adaptive_conflict_bonus'):
            base_obs.local_search_adaptive_conflict_bonus = max(0, int(adaptive_conflict_bonus))
        if hasattr(base_obs, 'local_search_adaptive_depth_bonus'):
            base_obs.local_search_adaptive_depth_bonus = max(0, int(adaptive_depth_bonus))
        if hasattr(base_obs, 'local_search_deadlock_probe_depth'):
            base_obs.local_search_deadlock_probe_depth = max(1, int(deadlock_probe_depth))
        if hasattr(base_obs, 'local_search_deadlock_max_states'):
            base_obs.local_search_deadlock_max_states = max(8, int(deadlock_max_states))
        if hasattr(base_obs, 'local_tree_clip_features'):
            base_obs.local_tree_clip_features = str(clip_tree_features).lower() == 'on'

    def _ctor_accepts_kwarg(cls, kwarg: str) -> bool:
        return kwarg in inspect.signature(cls.__init__).parameters

    def _build_temporal_obs(base_obs=None):
        kwargs = {'temporal_window': TEMPORAL_WINDOW}
        if base_obs is not None:
            kwargs['base_obs'] = base_obs
        if _ctor_accepts_kwarg(TemporalMultiAgentObservation, 'debug'):
            kwargs['debug'] = debug
        return TemporalMultiAgentObservation(**kwargs)

    if USE_HIERARCHICAL_OBS:
        hr_kwargs = {}
        if _ctor_accepts_kwarg(HierarchicalRoutesObservation, 'debug'):
            hr_kwargs['debug'] = debug
        if _ctor_accepts_kwarg(HierarchicalRoutesObservation, 'search_depth'):
            hr_kwargs['search_depth'] = search_depth
        base = HierarchicalRoutesObservation(**hr_kwargs)
        if not hr_kwargs and hasattr(base, 'search_depth'):
            base.search_depth = max(1, int(search_depth))
        _apply_tree_search_cfg(base)
        return _build_temporal_obs(base_obs=base)

    obs = _build_temporal_obs(base_obs=None)
    if hasattr(obs, 'base_obs'):
        _apply_tree_search_cfg(obs.base_obs)
    return obs


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
        # Increased entropy pressure for better exploration (done-rate too low at 0.095)
        weight_entropy=0.06,
        weight_value=0.5,
        # Keep auxiliary signal active but avoid overpowering PPO objective.
        weight_aux_dl=0.035,
        temporal_window=TEMPORAL_WINDOW,
        train_frequency=10,     # ⬆️ Train every 10 episodes (faster feedback)
        reward_scale=0.12,
        aux_pos_weight=4.0,
        target_kl=0.04,
        max_eps_random=0.02,
        clear_buffer_after_update=True,
    )
    policy.eps_smoothing = eps  # Set epsilon floor
    return policy


# Central speed profiles
default_hidden_size = 64
default_batch_size = 256
default_batch_fraction = 0.8
default_max_batches = 8
default_memory_episodes = 20

# Recommended default: stronger PPO update to avoid near-zero policy drift.
default_k_epochs = 3

# NOTE: ppo_param will be REBUILT after CLI args parsing (in main section)
# This version is only for non-main use (imports, testing)
ppo_param = MARL_ATTENTION_TEMPORAL_MAPPO_Param(
    hidden_size=_env_int('FLATLAND_HIDDEN_SIZE', default_hidden_size),
    batch_size=max(32, _env_int('FLATLAND_BATCH_SIZE', default_batch_size)),
    learning_rate=_env_float('FLATLAND_LR', 1.8e-5),
    discount=_env_float('FLATLAND_DISCOUNT', 0.99),
    gae_lambda=_env_float('FLATLAND_GAE_LAMBDA', 0.92),  # ↓ 0.95→0.92: sharper advantage signal
    use_gpu=USE_GPU_EFFECTIVE,
    max_episodes_in_training_memory=max(4, _env_int('FLATLAND_TRAIN_MEMORY_EPISODES', default_memory_episodes)),
    k_epochs=max(1, _env_int('FLATLAND_K_EPOCHS', default_k_epochs)),
    batch_fraction=min(1.0, max(0.2, _env_float('FLATLAND_BATCH_FRACTION', default_batch_fraction))),
    max_batches_per_training=max(1, _env_int('FLATLAND_MAX_BATCHES', default_max_batches)),
    temporal_window=TEMPORAL_WINDOW,
    encoder_type=os.getenv('FLATLAND_ENCODER_TYPE', 'lstm'),
    encoder_shared=os.getenv('FLATLAND_ENCODER_SHARED', 'false').lower() == 'true',
    use_spatial_attention=os.getenv('FLATLAND_USE_SPATIAL_ATTENTION', 'true').lower() == 'true'
)

def create_ma_ppo_agent(observation_space: int, action_space: int, eps: float = 0.0, optimizer_mode: str = 'single') -> LearningPolicy:
    """
    Creates PPO Policy with Temporal Transformer Encoder
    
    observation_space: temporal observation size (24D DecisionPoint or 48D HierarchicalRoutes)
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

def create_ma_ppo_agent_dp(observation_space: int, action_space: int, eps: float = 0.0, optimizer_mode: str = 'multiple') -> LearningPolicy:
    """
    Creates  PPO Policy with Temporal Transformer Encoder
    
    observation_space: temporal observation size (24D DecisionPoint or 48D HierarchicalRoutes)
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
        
    effective_ppo_param = ppo_param
    # Scalable-simple profile for CORE PPO runs: reduce architecture complexity
    # before scaling to many agents.
    simplified_val = str(os.getenv('FLATLAND_SIMPLIFIED_MAPPO', '0')).strip()
    try:
        simplified_n = int(simplified_val)
    except ValueError:
        simplified_n = 1 if simplified_val.lower() in ('1', 'true', 'yes', 'on') else 0
    use_scalable_simple = simplified_n > 0 and str(os.getenv('FLATLAND_SCALABLE_SIMPLE_PROFILE', '1')).strip().lower() in ('1', 'true', 'yes', 'on')
    if use_scalable_simple:
        effective_ppo_param = ppo_param._replace(encoder_shared=True, use_spatial_attention=False)

    policy = MARL_ATT_DecisionPointPolicy(
        observation_space,
        action_space,
        effective_ppo_param,
        show_pre_train_debug_msg=False,
        show_progress_bar=True,
        train_frequency=10,
        optimizer_mode=optimizer_mode,
        use_action_masking=str(os.getenv('FLATLAND_USE_ACTION_MASKING', '1')).strip().lower() in ('1', 'true', 'yes', 'on')
    )
    # Offensive baseline (default):
    # - prioritize progress/forward flow
    # - keep stronger decision-point exploration to avoid deadlock plateaus
    # - reduce forward-only collapse while preserving throughput bias
    policy.surrogate_eps_clip = float(np.clip(_env_float('FLATLAND_CLIP_EPS', 0.18), 0.05, 0.40))
    policy.weight_entropy = float(np.clip(_env_float('FLATLAND_WEIGHT_ENTROPY', 0.12), 0.0, 1.0))  # ↑ more exploration
    policy.reward_scale = 0.09
    policy.weight_loss = float(np.clip(_env_float('FLATLAND_WEIGHT_VALUE', 1.10), 0.1, 5.0))      # ↓ less critic pressure
    policy.stability_guard_start_episode = 1200
    policy.stability_guard_hard_episode = 2600
    policy.ppo_target_kl = 0.040
    policy.ppo_max_kl = 0.080
    policy.ppo_emergency_kl = 0.16
    policy.ppo_emergency_kl_hard = 0.24
    policy.ratio_guard_soft = 1.14
    policy.ratio_guard_soft_low = 0.86
    policy.ratio_guard_hard = 1.22
    policy.ratio_guard_hard_low = 0.78
    policy.max_hard_batches_before_lr_decay = 4
    policy.hard_spike_streak_limit = 4
    policy.actor_lr_min_factor = 0.70
    policy.actor_lr_decay_on_instability = 0.88
    # Stronger default exploration for deadlock-heavy decision-point regimes.
    policy.max_eps_random = float(np.clip(_env_float('FLATLAND_MAX_EPS_RANDOM', 0.22), 0.0, 1.0)) # ↑ more random actions
    policy.decision_eps_floor = float(np.clip(_env_float('FLATLAND_DECISION_EPS_FLOOR', 0.22), 0.0, 1.0)) # ↑ more random actions
    policy.use_decision_eps_floor = str(os.getenv('FLATLAND_USE_DECISION_EPS_FLOOR', '1')).strip().lower() in ('1', 'true', 'yes', 'on')
    # Keep some turning signal and activate diversity shaping earlier at decision points.
    policy.weight_action_diversity = 0.24
    policy.action_diversity_gate_threshold = 0.30
    policy.forward_prob_soft_max = 0.70
    policy.lr_prob_soft_min = 0.08
    policy.idle_prob_soft_max = 0.10
    policy.idle_logit_penalty = 1.60
    policy.stop_logit_penalty = 1.10

    # For single-agent core runs, prioritize fast convergence over exploration.
    # This avoids a persistent ~20% random-failure floor caused by high epsilon floors.
    if simplified_n == 1:
        policy.use_decision_eps_floor = str(os.getenv('FLATLAND_USE_DECISION_EPS_FLOOR', '0')).strip().lower() in ('1', 'true', 'yes', 'on')
        policy.decision_eps_floor = float(np.clip(_env_float('FLATLAND_DECISION_EPS_FLOOR', 0.00), 0.0, 1.0))
        policy.max_eps_random = float(np.clip(_env_float('FLATLAND_MAX_EPS_RANDOM', 0.05), 0.0, 1.0))
        policy.sp_hint_route_prior_prob = float(np.clip(_env_float('FLATLAND_SP_HINT_ROUTE_PRIOR_PROB', 0.92), 0.0, 1.0))
        policy.sp_hint_logit_bonus = float(np.clip(_env_float('FLATLAND_SP_HINT_LOGIT_BONUS', 1.80), 0.0, 4.0))
        print('   - single-agent convergence mode: lower exploration, stronger SP prior')

    policy.eps_smoothing = eps  # Set epsilon floor
    print(
        f"   - exploration: decision_eps_floor={policy.decision_eps_floor:.3f}, "
        f"max_eps_random={policy.max_eps_random:.3f}, "
        f"use_decision_eps_floor={bool(policy.use_decision_eps_floor)}, "
        f"weight_entropy={policy.weight_entropy:.3f}, "
        f"clip_eps={policy.surrogate_eps_clip:.3f}, "
        f"weight_value={policy.weight_loss:.3f}"
    )
    print(
        f"   - route_prior: sp_prob={policy.sp_hint_route_prior_prob:.2f}, "
        f"sp_logit_bonus={policy.sp_hint_logit_bonus:.2f}"
    )
    if use_scalable_simple:
        print('   - scalable_simple_profile: encoder_shared=True, use_spatial_attention=False')
    print('   - profile: OFFENSIVE_BASELINE_V2 (anti-deadlock tuned)')
    return policy


def create_ma_ppo_agent_dp_dla(observation_space: int, action_space: int, eps: float = 0.0, optimizer_mode: str = 'multiple') -> LearningPolicy:
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
        epilog="""
QUICK START EXAMPLES:
  # Default (robust + balanced speed): separate encoders, spatial attention enabled
  python marl_attention_temporal.py final --eps 0.0
  
  # Optimized for CPU (fast + robust): shared encoder, spatial attention enabled  
  python marl_attention_temporal.py final --eps 0.0 --encoder-shared
  
  # Lightweight (very fast): shared encoder, NO spatial attention
  python marl_attention_temporal.py final --eps 0.0 --encoder-shared --no-spatial-attention
  
  # Continue training from checkpoint  
  python marl_attention_temporal.py final_continue --eps 0.1 --encoder-shared
  
  # Evaluation with optimized architecture
  python marl_attention_temporal.py --eval --encoder-shared

  #simplified single-agent test (debugging, architecture ablation, etc.) 
  FLATLAND_SIMPLIFIED_MAPPO=o python marl_attention_temporal.py new --DEBUG # off
  FLATLAND_SIMPLIFIED_MAPPO=1 python marl_attention_temporal.py new --DEBUG # 1 agent
  FLATLAND_SIMPLIFIED_MAPPO=5 python marl_attention_temporal.py new --DEBUG # 5 agents (full mode)
  FLATLAND_SIMPLIFIED_MAPPO=10 python marl_attention_temporal.py new --DEBUG # 10 agents (stress test)
  
ARCHITECTURE CONFIGURATION:
  Default: encoder_shared=False, use_spatial_attention=True
    ✅ Full model capacity (best for complex coordination)
    ✅ Multi-agent spatial attention (MAAC-style)
    ⚡ Medium speed (2× encoder forward passes)
    
  Recommended for CPU: --encoder-shared
    ✅ Keeps spatial attention (multi-agent learning)
    ⚡ 50% faster (~50% fewer parameters)
    
  Maximum speed: --encoder-shared --no-spatial-attention  
    ⚡ 70% faster overall
    ⚠️  Loses agent-agent attention (temporal-only)
    
  For GPU: Keep defaults (separate encoders, spatial attention)

EXAMPLES WITH TREE SEARCH TUNING:
  # Fast tree search + shared encoder
  python marl_attention_temporal.py final --eps 0.0 --encoder-shared --search_depth 8
  
  # Minimal tree search + maximum speed optimization
  python marl_attention_temporal.py final --eps 0.0 --encoder-shared --no-spatial-attention --search_depth 6

LEGACY EXAMPLES (still supported):
  python marl_attention_temporal.py --train --fresh-start
  python marl_attention_temporal.py --train --continue --eps 0.1
  python marl_attention_temporal.py --eval
    # Default (robust, balanced speed)
    python marl_attention_temporal.py final --eps 0.0

    # CPU-optimized (~50% faster) 
    python marl_attention_temporal.py final --eps 0.0 --encoder-shared

    # Maximum speed (~70% faster, temporal-only)
    python marl_attention_temporal.py final --eps 0.0 --encoder-shared --no-spatial-attention
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
        default=0.03,
        metavar='EPS_VALUE',
        dest='eps',
        help='Exploration floor for epsilon-greedy in [0.0, 1.0] (default: 0.03)'
    )

    parser.add_argument(
        '--min_eps',
        type=float,
        default=None,
        metavar='MIN_EPS_VALUE',
        dest='min_eps',
        help='Minimum exploration floor for epsilon-greedy in [0.0, 1.0] (default: 0.01; with --eps 0.0 and no --min_eps, floor is disabled)'
    )

    parser.add_argument(
        '--optimizer_mode',
        type=str,
        default='single',
        choices=['single', 'multiple'],
        metavar='MODE',
        dest='optimizer_mode',
        help='Optimizer mode: single = consolidated single optimizer, multiple = 4 optimizers with synchronized decay and stronger critic defaults (default: single)'
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
    parser.add_argument(
        '--search_depth',
        type=int,
        default=LOCAL_TREE_SEARCH_DEPTH,
        metavar='DEPTH',
        dest='search_depth',
        help='Local tree search depth for observation builder (default: 12; recommended 8-12 with adaptive budgeting)'
    )
    parser.add_argument(
        '--tree_random_start_depth',
        type=int,
        default=LOCAL_TREE_RANDOM_START_DEPTH,
        metavar='DEPTH',
        dest='tree_random_start_depth',
        help='From this tree depth onward, side branches are sampled (default: 2)'
    )
    parser.add_argument(
        '--tree_max_side_branches',
        type=int,
        default=LOCAL_TREE_MAX_SIDE_BRANCHES,
        metavar='K',
        dest='tree_max_side_branches',
        help='Maximum sampled side branches per node after shortest branch (default: 1)'
    )
    parser.add_argument(
        '--tree_distance_bias',
        type=float,
        default=LOCAL_TREE_DISTANCE_BIAS,
        metavar='ALPHA',
        dest='tree_distance_bias',
        help='Sampling bias toward shorter side branches; larger means stronger short-path bias (default: 2.0)'
    )
    parser.add_argument(
        '--tree_mode',
        type=str,
        default=LOCAL_TREE_MODE,
        choices=['stochastic', 'mcts'],
        metavar='MODE',
        dest='tree_mode',
        help='Branch-selection mode after tree_random_start_depth: stochastic or mcts (default: stochastic)'
    )
    parser.add_argument(
        '--tree_mcts_rollouts',
        type=int,
        default=LOCAL_TREE_MCTS_ROLLOUTS,
        metavar='N',
        dest='tree_mcts_rollouts',
        help='MCTS-lite rollout budget per expanded node when tree_mode=mcts (default: 6)'
    )
    parser.add_argument(
        '--tree_mcts_horizon',
        type=int,
        default=LOCAL_TREE_MCTS_HORIZON,
        metavar='H',
        dest='tree_mcts_horizon',
        help='Rollout horizon in rail cells for tree_mode=mcts (default: 4)'
    )
    parser.add_argument(
        '--tree_ucb_c',
        type=float,
        default=LOCAL_TREE_UCB_C,
        metavar='C',
        dest='tree_ucb_c',
        help='Exploration constant for MCTS-lite UCB action selection (default: 1.2)'
    )
    parser.add_argument(
        '--tree_contract_depth',
        type=int,
        default=LOCAL_TREE_CONTRACT_DEPTH,
        metavar='DEPTH',
        dest='tree_contract_depth',
        help='From this depth onward linear corridors are contracted into one edge (default: 8)'
    )
    parser.add_argument(
        '--tree_max_nodes',
        type=int,
        default=LOCAL_TREE_MAX_NODES,
        metavar='N',
        dest='tree_max_nodes',
        help='Hard node budget for local tree search per agent step (default: 72)'
    )
    parser.add_argument(
        '--tree_min_nodes',
        type=int,
        default=LOCAL_TREE_MIN_NODES,
        metavar='N',
        dest='tree_min_nodes',
        help='Minimum node budget when adaptive budgeting is enabled (default: 32)'
    )
    parser.add_argument(
        '--tree_adaptive_budget',
        type=str,
        default=LOCAL_TREE_ADAPTIVE_BUDGET,
        choices=['on', 'off'],
        metavar='MODE',
        dest='tree_adaptive_budget',
        help='Adaptive node budget mode: on lowers cost in simple scenes, off uses fixed max_nodes (default: on)'
    )
    parser.add_argument(
        '--tree_adaptive_branch_bonus',
        type=int,
        default=LOCAL_TREE_ADAPTIVE_BRANCH_BONUS,
        metavar='N',
        dest='tree_adaptive_branch_bonus',
        help='Node bonus per extra root branch for adaptive budgeting (default: 6)'
    )
    parser.add_argument(
        '--tree_adaptive_conflict_bonus',
        type=int,
        default=LOCAL_TREE_ADAPTIVE_CONFLICT_BONUS,
        metavar='N',
        dest='tree_adaptive_conflict_bonus',
        help='Node bonus for merge/conflict hotspots in adaptive budgeting (default: 14)'
    )
    parser.add_argument(
        '--tree_adaptive_depth_bonus',
        type=int,
        default=LOCAL_TREE_ADAPTIVE_DEPTH_BONUS,
        metavar='N',
        dest='tree_adaptive_depth_bonus',
        help='Node bonus per depth step above 6 in adaptive budgeting (default: 2)'
    )
    parser.add_argument(
        '--tree_deadlock_probe_depth',
        type=int,
        default=LOCAL_TREE_DEADLOCK_PROBE_DEPTH,
        metavar='DEPTH',
        dest='tree_deadlock_probe_depth',
        help='Depth cap for per-node deadlock probe used inside local search (default: 7)'
    )
    parser.add_argument(
        '--tree_deadlock_max_states',
        type=int,
        default=LOCAL_TREE_DEADLOCK_MAX_STATES,
        metavar='N',
        dest='tree_deadlock_max_states',
        help='State cap for per-node deadlock probe used inside local search (default: 96)'
    )
    parser.add_argument(
        '--tree_clip_features',
        type=str,
        default=LOCAL_TREE_CLIP_FEATURES,
        choices=['on', 'off'],
        metavar='MODE',
        dest='tree_clip_features',
        help='Clip serialized tree-node features to [0,1] before policy input (default: on)'
    )
    
    # ====================================================================
    # ARCHITECTURE OPTIMIZATION FLAGS
    # ====================================================================
    # DEFAULT STRATEGY: encoder_shared=False, use_spatial_attention=True
    # ✅ Robust: Full model capacity per head
    # ✅ Scalable: Multi-agent coordination learned via spatial attention
    # ⚡ Optimize with --encoder-shared for CPU-bound training (50% faster)
    # ⚡ Optimize with --no-spatial-attention for single-agent or temporal-only (20% faster)
    parser.add_argument(
        '--encoder-shared',
        action='store_true',
        dest='encoder_shared',
        default=True,
        help='Share single encoder between actor+critic (~50% faster, -50% params). Default: True (shared encoder for speed).'
    )
    parser.add_argument(
        '--no-spatial-attention',
        action='store_true',
        dest='no_spatial_attention',
        help='Disable spatial attention (agent×opponent). (~20% faster, temporal-only). Default: False (spatial attention enabled for multi-agent learning)'
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
    min_eps_user_set = args.min_eps is not None
    min_eps = float(args.min_eps) if min_eps_user_set else 0.02
    optimizer_mode = args.optimizer_mode.upper()
    rendering = bool(args.rendering)
    policy_mode = args.policy_mode.strip().lower()
    debug_mode = args.debug  # Capture debug flag
    
    # ====================================================================
    # ARCHITECTURE FLAGS from CLI
    # ====================================================================
    encoder_shared_cli = True  # Default: True (shared encoder for speed)
    use_spatial_attention_cli = not bool(args.no_spatial_attention)  # Default: True (spatial attention enabled)
    
    search_depth = int(args.search_depth)
    tree_random_start_depth = int(args.tree_random_start_depth)
    tree_max_side_branches = int(args.tree_max_side_branches)
    tree_distance_bias = float(args.tree_distance_bias)
    tree_mode = str(args.tree_mode).lower()
    tree_mcts_rollouts = int(args.tree_mcts_rollouts)
    tree_mcts_horizon = int(args.tree_mcts_horizon)
    tree_ucb_c = float(args.tree_ucb_c)
    tree_contract_depth = int(args.tree_contract_depth)
    tree_max_nodes = int(args.tree_max_nodes)
    tree_min_nodes = int(args.tree_min_nodes)
    tree_adaptive_budget = str(args.tree_adaptive_budget).lower()
    tree_adaptive_branch_bonus = int(args.tree_adaptive_branch_bonus)
    tree_adaptive_conflict_bonus = int(args.tree_adaptive_conflict_bonus)
    tree_adaptive_depth_bonus = int(args.tree_adaptive_depth_bonus)
    tree_deadlock_probe_depth = int(args.tree_deadlock_probe_depth)
    tree_deadlock_max_states = int(args.tree_deadlock_max_states)
    tree_clip_features = str(args.tree_clip_features).lower()
    do_training = mode != 'eval'
    do_rendering = rendering
    checkpoint_interval = 50
    start_from_phase = 0
    
    # ====================================================================
    # REBUILD PPO PARAMETERS with CLI arguments (override env vars)
    # ====================================================================
    # This replaces the global ppo_param with CLI-configured version
    ppo_param = MARL_ATTENTION_TEMPORAL_MAPPO_Param(
        hidden_size=_env_int('FLATLAND_HIDDEN_SIZE', default_hidden_size),
        batch_size=max(32, _env_int('FLATLAND_BATCH_SIZE', default_batch_size)),  # Use mode-aware default (FAST:128, else:256)
        learning_rate=_env_float('FLATLAND_LR', 1.8e-5),
        discount=_env_float('FLATLAND_DISCOUNT', 0.99),
        gae_lambda=_env_float('FLATLAND_GAE_LAMBDA', 0.95),
        use_gpu=USE_GPU_EFFECTIVE,
        max_episodes_in_training_memory=max(4, _env_int('FLATLAND_TRAIN_MEMORY_EPISODES', default_memory_episodes)),
        k_epochs=max(1, _env_int('FLATLAND_K_EPOCHS', default_k_epochs)),
        batch_fraction=min(1.0, max(0.2, _env_float('FLATLAND_BATCH_FRACTION', default_batch_fraction))),
        max_batches_per_training=max(1, _env_int('FLATLAND_MAX_BATCHES', default_max_batches)),
        temporal_window=TEMPORAL_WINDOW,
        encoder_type=os.getenv('FLATLAND_ENCODER_TYPE', 'lstm'),
        encoder_shared=True,  # Immer shared encoder für Speed
        use_spatial_attention=use_spatial_attention_cli  # Spatial Attention bleibt steuerbar
    )
    
    if USE_CURRICULUM_PHASES and mode in ('final', 'final_continue'):
        phase_idx_by_name = {p['name']: idx for idx, p in enumerate(CURRICULUM_PHASES)}
        if 'phase5_final' in phase_idx_by_name:
            start_from_phase = phase_idx_by_name['phase5_final']
        else:
            start_from_phase = max(0, len(CURRICULUM_PHASES) - 1)
            print("[Warn] phase5_final not found in CURRICULUM_PHASES. Falling back to last phase.")

    # Validate EPS range
    if not (0.0 <= eps <= 1.0):
        print(f"ERROR: --eps must be between 0.0 and 1.0, got {eps}")
        sys.exit(1)
    if not (0.0 <= min_eps <= 1.0):
        print(f"ERROR: --min_eps must be between 0.0 and 1.0, got {min_eps}")
        sys.exit(1)
    if do_training and eps <= 0.0 and not min_eps_user_set:
        min_eps = 0.03
        print("[Info] eps=0.0 in training and --min_eps not set: using rescue epsilon floor (min_eps=0.03).")
    elif do_training and eps <= 0.0:
        if min_eps <= 0.0:
            print("[Warn] eps=0.0 and min_eps=0.0 in training: global epsilon exploration is disabled.")
        else:
            print(f"[Info] eps=0.0 in training: using min_eps floor={min_eps:.4f} for global exploration.")
    if not (1 <= search_depth <= 12):
        print(f"ERROR: --search_depth must be between 1 and 12, got {search_depth}")
        sys.exit(1)
    if not (0 <= tree_random_start_depth <= 12):
        print(f"ERROR: --tree_random_start_depth must be between 0 and 12, got {tree_random_start_depth}")
        sys.exit(1)
    if not (0 <= tree_max_side_branches <= 3):
        print(f"ERROR: --tree_max_side_branches must be between 0 and 3, got {tree_max_side_branches}")
        sys.exit(1)
    if not (0.1 <= tree_distance_bias <= 10.0):
        print(f"ERROR: --tree_distance_bias must be between 0.1 and 10.0, got {tree_distance_bias}")
        sys.exit(1)
    if tree_mode not in ('stochastic', 'mcts'):
        print(f"ERROR: --tree_mode must be one of ['stochastic', 'mcts'], got {tree_mode}")
        sys.exit(1)
    if not (1 <= tree_mcts_rollouts <= 64):
        print(f"ERROR: --tree_mcts_rollouts must be between 1 and 64, got {tree_mcts_rollouts}")
        sys.exit(1)
    if not (1 <= tree_mcts_horizon <= 16):
        print(f"ERROR: --tree_mcts_horizon must be between 1 and 16, got {tree_mcts_horizon}")
        sys.exit(1)
    if not (0.01 <= tree_ucb_c <= 4.0):
        print(f"ERROR: --tree_ucb_c must be between 0.01 and 4.0, got {tree_ucb_c}")
        sys.exit(1)
    if not (0 <= tree_contract_depth <= 12):
        print(f"ERROR: --tree_contract_depth must be between 0 and 12, got {tree_contract_depth}")
        sys.exit(1)
    if not (8 <= tree_max_nodes <= 256):
        print(f"ERROR: --tree_max_nodes must be between 8 and 256, got {tree_max_nodes}")
        sys.exit(1)
    if not (8 <= tree_min_nodes <= 256):
        print(f"ERROR: --tree_min_nodes must be between 8 and 256, got {tree_min_nodes}")
        sys.exit(1)
    if tree_min_nodes > tree_max_nodes:
        print(f"ERROR: --tree_min_nodes must be <= --tree_max_nodes, got {tree_min_nodes}>{tree_max_nodes}")
        sys.exit(1)
    if tree_adaptive_budget not in ('on', 'off'):
        print(f"ERROR: --tree_adaptive_budget must be one of ['on', 'off'], got {tree_adaptive_budget}")
        sys.exit(1)
    if not (0 <= tree_adaptive_branch_bonus <= 32):
        print(f"ERROR: --tree_adaptive_branch_bonus must be between 0 and 32, got {tree_adaptive_branch_bonus}")
        sys.exit(1)
    if not (0 <= tree_adaptive_conflict_bonus <= 32):
        print(f"ERROR: --tree_adaptive_conflict_bonus must be between 0 and 32, got {tree_adaptive_conflict_bonus}")
        sys.exit(1)
    if not (0 <= tree_adaptive_depth_bonus <= 16):
        print(f"ERROR: --tree_adaptive_depth_bonus must be between 0 and 16, got {tree_adaptive_depth_bonus}")
        sys.exit(1)
    if not (1 <= tree_deadlock_probe_depth <= 32):
        print(f"ERROR: --tree_deadlock_probe_depth must be between 1 and 32, got {tree_deadlock_probe_depth}")
        sys.exit(1)
    if not (8 <= tree_deadlock_max_states <= 512):
        print(f"ERROR: --tree_deadlock_max_states must be between 8 and 512, got {tree_deadlock_max_states}")
        sys.exit(1)
    if tree_clip_features not in ('on', 'off'):
        print(f"ERROR: --tree_clip_features must be one of ['on', 'off'], got {tree_clip_features}")
        sys.exit(1)
    # Honor the configured minimum exploration floor even when --eps is 0.0.
    if do_training and min_eps > eps:
        eps = float(min_eps)

    print(
        f"\n[Config] mode={mode}, eps={eps:.4f}, optimizer_mode={optimizer_mode}, "
        f"encoder_shared={encoder_shared_cli}, use_spatial_attention={use_spatial_attention_cli}, "
        f"policy_mode={policy_mode}, debug={debug_mode}, search_depth={search_depth}, "
        f"tree_mode={tree_mode}, tree_start={tree_random_start_depth}, tree_k={tree_max_side_branches}, "
        f"tree_bias={tree_distance_bias:.2f}, tree_rollouts={tree_mcts_rollouts}, "
        f"tree_horizon={tree_mcts_horizon}, tree_ucb_c={tree_ucb_c:.2f}, "
        f"tree_contract_depth={tree_contract_depth}, tree_min_nodes={tree_min_nodes}, "
        f"tree_max_nodes={tree_max_nodes}, tree_adaptive_budget={tree_adaptive_budget}, "
        f"tree_adaptive_branch_bonus={tree_adaptive_branch_bonus}, "
        f"tree_adaptive_conflict_bonus={tree_adaptive_conflict_bonus}, "
        f"tree_adaptive_depth_bonus={tree_adaptive_depth_bonus}, "
        f"tree_deadlock_probe_depth={tree_deadlock_probe_depth}, tree_deadlock_max_states={tree_deadlock_max_states}, "
        f"tree_clip_features={tree_clip_features}"
    )
    if USE_CURRICULUM_PHASES:
        print(f"[Config] curriculum_start_phase_index={start_from_phase} ({CURRICULUM_PHASES[start_from_phase]['name']})")

    environment = RailEnvironmentPersistable(
        obs_builder_object_creator=lambda: create_temporal_obs_builder_object(
            debug=debug_mode,
            search_depth=search_depth,
            random_start_depth=tree_random_start_depth,
            max_side_branches=tree_max_side_branches,
            distance_bias=tree_distance_bias,
            tree_mode=tree_mode,
            mcts_rollouts=tree_mcts_rollouts,
            mcts_horizon=tree_mcts_horizon,
            ucb_c=tree_ucb_c,
            contract_depth=tree_contract_depth,
            max_nodes=tree_max_nodes,
            min_nodes=tree_min_nodes,
            adaptive_budget=tree_adaptive_budget,
            adaptive_branch_bonus=tree_adaptive_branch_bonus,
            adaptive_conflict_bonus=tree_adaptive_conflict_bonus,
            adaptive_depth_bonus=tree_adaptive_depth_bonus,
            deadlock_probe_depth=tree_deadlock_probe_depth,
            deadlock_max_states=tree_deadlock_max_states,
            clip_tree_features=tree_clip_features,
        ),
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
    _obs_builder_for_size = create_temporal_obs_builder_object(
        search_depth=search_depth,
        random_start_depth=tree_random_start_depth,
        max_side_branches=tree_max_side_branches,
        distance_bias=tree_distance_bias,
        tree_mode=tree_mode,
        mcts_rollouts=tree_mcts_rollouts,
        mcts_horizon=tree_mcts_horizon,
        ucb_c=tree_ucb_c,
        contract_depth=tree_contract_depth,
        max_nodes=tree_max_nodes,
        min_nodes=tree_min_nodes,
        adaptive_budget=tree_adaptive_budget,
        adaptive_branch_bonus=tree_adaptive_branch_bonus,
        adaptive_conflict_bonus=tree_adaptive_conflict_bonus,
        adaptive_depth_bonus=tree_adaptive_depth_bonus,
        deadlock_probe_depth=tree_deadlock_probe_depth,
        deadlock_max_states=tree_deadlock_max_states,
        clip_tree_features=tree_clip_features,
    )
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
        # Keep runtime epsilon bounded by the CLI value unless explicitly
        # re-enabled in a policy experiment.
        policy.cli_eps_cap = float(eps)
        policy.allow_eps_above_cli = False
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


        solver.set_reward_shaper(
            FlatlandSparseRewardShaper(
                    step_penalty=REWARD_STEP_PENALTY,
                    done_bonus=REWARD_DONE_BONUS,
                    all_done_bonus=REWARD_ALL_DONE_BONUS,
                    deadlock_penalty=REWARD_DEADLOCK_PENALTY,
                    progress_bonus=REWARD_PROGRESS_BONUS,
                    final_not_solved_penalty=FINAL_NOT_SOLVED_PENALTY
                )
        )
 
        latest_ckpt = f"training_output/last_checkpoint/{solver.get_name()}_{solver.policy.get_name()}"
        aux_weight_fields = []
        for _field in ("weight_aux_deadlock", "weight_aux_dl"):
            if hasattr(policy, _field):
                try:
                    aux_weight_fields.append((_field, float(getattr(policy, _field))))
                except Exception:
                    pass

        exploration_fields = []
        for _field in ("decision_eps_floor", "max_eps_random"):
            if hasattr(policy, _field):
                try:
                    exploration_fields.append((_field, float(getattr(policy, _field))))
                except Exception:
                    pass
        
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

                    is_single_agent_phase = len(phase_agents) == 1 and int(phase_agents[0]) == 1
                    phase_eps = float(eps)
                    phase_min_eps = float(min_eps)

                    if is_single_agent_phase:
                        phase_eps = min(phase_eps, 0.08)
                        phase_min_eps = min(phase_min_eps, 0.02)
                        if phase_min_eps > phase_eps:
                            phase_eps = phase_min_eps

                    for _field, _base in aux_weight_fields:
                        try:
                            setattr(policy, _field, 0.0 if is_single_agent_phase else _base)
                        except Exception:
                            pass

                    for _field, _base in exploration_fields:
                        try:
                            if is_single_agent_phase:
                                setattr(policy, _field, min(_base, 0.05))
                            else:
                                setattr(policy, _field, _base)
                        except Exception:
                            pass

                    print(
                        f"[Train] {phase['name']}: {phase_episodes} episodes, agents={phase_agents}, "
                        f"eps={phase_eps:.3f}, min_eps={phase_min_eps:.3f}, "
                        f"aux_deadlock={'off' if is_single_agent_phase else 'on'}"
                    )
                    solver.perform_training(
                        max_episodes=phase_episodes,
                        checkpoint_interval=checkpoint_interval,
                        eps=phase_eps,
                        min_eps=phase_min_eps,
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
            
