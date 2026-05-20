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
# [8] MAPPO official GitHub repository (reference implementation):
#     https://github.com/marlbenchmark/on-policy
# [9] CleanRL GitHub repository (PPO implementation details and baselines):
#     https://github.com/vwxyzjn/cleanrl
# [10] Flatland-RL GitHub repository (environment and task domain codebase):
#      https://github.com/flatland-association/flatland-rl
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
from marl_attention_temporal_observation.decision_point_utils import DecisionPointUtils  # noqa: E402

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


REWARD_STEP_PENALTY = _env_float('FLATLAND_REWARD_STEP_PENALTY', -0.01)
REWARD_DONE_BONUS = _env_float('FLATLAND_REWARD_DONE_BONUS', 1.0)  # Boosted for stronger positive signal
REWARD_ALL_DONE_BONUS = _env_float('FLATLAND_REWARD_ALL_DONE_BONUS', 10.0)
REWARD_DEADLOCK_PENALTY = _env_float('FLATLAND_DEADLOCK_PENALTY', -5.0)   # Massiv gesenkt: 100→8, verhindert Reward-Varianz-Explosion
REWARD_PROGRESS_BONUS = _env_float('FLATLAND_REWARD_PROGRESS_BONUS', 0.01)  # ↑ reward for progress
FINAL_NOT_SOLVED_PENALTY = _env_float('FLATLAND_FINAL_NOT_SOLVED_PENALTY', -2.0)   # Gesenkt: 10→2, reduziert Varianz der Rückgabe

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
        near_step_limit = raw_env._elapsed_steps > (raw_env._max_episode_steps - 5)
        active_on_map_handles = [
            int(a.handle)
            for a in raw_env.agents
            if a.state != TrainState.DONE and a.position is not None and a.direction is not None
        ]
        deadlock_check_enabled = len(active_on_map_handles) > 1
        give_all_done_bonus = all_agents_done and not self._all_done_bonus_given
 
        for handle in env.get_agent_handles():
            agent = raw_env.agents[handle]
            
            r = 0.0
                
            if agent.state > TrainState.WAITING:
                # Apply time pressure for every non-terminal agent so idling is costly.
                if agent.state < TrainState.DONE:
                    r = self.step_penalty

                    if agent.position is not None and agent.direction is not None:
                        current_dist = self._current_agent_distance(env, agent)
                        prev_dist = float(self._prev_distance.get(handle, current_dist))
                        if current_dist < prev_dist:
                            r = self.progress_bonus
                        self._prev_distance[handle] = current_dist

                        if deadlock_check_enabled and DecisionPointUtils.is_local_deadlock(raw_env, agent, agent_map):
                            self._current_episode_deadlocks.add(int(handle))
                            r = self.deadlock_penalty

                # +BONUS once when an agent reaches target.
                if agent.state == TrainState.DONE and not bool(self._rewarded_done.get(handle, False)):
                    r = self.done_bonus
                    self._rewarded_done[handle] = True
 
                # If all agents are done, grant one-time team bonus to each agent.
                if give_all_done_bonus:
                    r = self.all_done_bonus

                # Final not solved penalty only when the episode is near step limit
                # and not fully solved. Keep all-done bonus intact in solved episodes.
                if near_step_limit and not all_agents_done and agent.state < TrainState.DONE:
                    r = self.final_not_solved_penalty

            shaped[handle] = float(r)

        if give_all_done_bonus:
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
            _env_float('FLATLAND_SP_HINT_ROUTE_PRIOR_PROB', 0.35),
            0.0,
            1.0,
        ))
        self.sp_hint_logit_bonus = float(np.clip(
            _env_float('FLATLAND_SP_HINT_LOGIT_BONUS', 0.35),
            0.0,
            4.0,
        ))

    def get_name(self):
        if self.use_deadlock_avoidance_policy:
            return self.__class__.__name__ + "_DLA"
        return self.__class__.__name__


    def step(self, handle, state, action, reward, next_state, done, agent_finished=None):
        super(MARL_ATT_DecisionPointPolicy, self).step(
            handle,
            state,
            action,
            reward,
            next_state,
            done,
            agent_finished=agent_finished,
        )
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
            if cell_type == 'MERGING' and action == RailEnvActions.MOVE_FORWARD:
                return self.deadlock_avoidance_policy.act(handle, state, eps)

        return action
 

# =============================================================================
# ENVIRONMENT & TRAINING SETUP
# =============================================================================

# Temporal context configuration.
# Full baseline default: temporal context enabled (window from env, default 3).
# Set FLATLAND_DISABLE_TEMPORAL_ATTENTION=1 to force stateless mode.
_DISABLE_TEMPORAL_ATTENTION = str(
    os.getenv('FLATLAND_DISABLE_TEMPORAL_ATTENTION', '0')
).strip().lower() in ('1', 'true', 'yes', 'on')

TEMPORAL_WINDOW = 1 if _DISABLE_TEMPORAL_ATTENTION else max(
    1,
    _env_int('FLATLAND_TEMPORAL_WINDOW', 3),
)




# ========================================================================
# DYNAMIC AGENT COUNT CONFIG: Override via FLATLAND_SIMPLIFIED_MAPPO env var
# ========================================================================
# FLATLAND_SIMPLIFIED_MAPPO=0           → standard 5-agent config
# FLATLAND_SIMPLIFIED_MAPPO=N           → fixed N-agent config
# FLATLAND_SIMPLIFIED_MAPPO=1,5,10,100  → agent-count sweep in one run
# NOTE: Core/Full mode is configured separately via FLATLAND_CORE_PPO_MODE.
# ========================================================================
simplified_agent_count = os.getenv('FLATLAND_SIMPLIFIED_MAPPO', '0').strip()
try:
    _simplified_n = int(simplified_agent_count)
    if _simplified_n > 0:
        PURE_MARL_AGENT_COUNTS = [_simplified_n]
        print(f"[Config] AGENT COUNT OVERRIDE: {_simplified_n} agent(s) (from FLATLAND_SIMPLIFIED_MAPPO)")
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
        print(f"[Config] AGENT COUNT SWEEP: agents={PURE_MARL_AGENT_COUNTS} (from FLATLAND_SIMPLIFIED_MAPPO)")
    else:
        PURE_MARL_AGENT_COUNTS = [5]  # Default full mode

# High-success curriculum: bias training toward hard coordination cases
# while keeping a small share of easy cases for stability.
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



def create_temporal_obs_builder_object(debug: bool = False):
    """
    Build TemporalMultiAgentObservation with strict base contract.
    All tree-search parameters are now managed exclusively in DecisionPointObservation.
    """
    def _ctor_accepts_kwarg(cls, kwarg: str) -> bool:
        import inspect
        return kwarg in inspect.signature(cls.__init__).parameters

    def _build_temporal_obs():
        kwargs = {'temporal_window': TEMPORAL_WINDOW}
        if _ctor_accepts_kwarg(TemporalMultiAgentObservation, 'debug'):
            kwargs['debug'] = debug
        return TemporalMultiAgentObservation(**kwargs)

    # Nur noch DecisionPointObservation als Basis
    obs = _build_temporal_obs()
    return obs



# Central speed profiles
default_hidden_size = 64
# PPO update budget: `batch_size * max_batches_per_training`.
# This keeps replay sampling independent of total buffer size.
# References: PPO (https://arxiv.org/abs/1707.06347), MAPPO (https://arxiv.org/abs/2103.01955),
# and implementation guidance (https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).
default_batch_size = 128
default_batch_fraction = 1.0
default_max_batches = 5
# Smaller on-policy window keeps updates fresher in sparse-done phases.
default_memory_episodes = 50

# Recommended default: stronger PPO update to avoid near-zero policy drift.
default_k_epochs = 3

# NOTE: ppo_param will be REBUILT after CLI args parsing (in main section)
# This version is only for non-main use (imports, testing)
ppo_param = MARL_ATTENTION_TEMPORAL_MAPPO_Param(
    hidden_size=_env_int('FLATLAND_HIDDEN_SIZE', default_hidden_size),
    batch_size=max(32, _env_int('FLATLAND_BATCH_SIZE', default_batch_size)),
    learning_rate=_env_float('FLATLAND_LR', 1.0e-5),
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
        simplified_n = 0

    core_mode_enabled = str(os.getenv('FLATLAND_CORE_PPO_MODE', '0')).strip().lower() in ('1', 'true', 'yes', 'on')
    use_scalable_simple = core_mode_enabled and str(os.getenv('FLATLAND_SCALABLE_SIMPLE_PROFILE', '1')).strip().lower() in ('1', 'true', 'yes', 'on')
    if use_scalable_simple:
        effective_ppo_param = ppo_param._replace(encoder_shared=True, use_spatial_attention=False)

    train_frequency_default = int(np.clip(_env_int('FLATLAND_TRAIN_FREQUENCY', 10), 1, 100))

    policy = MARL_ATT_DecisionPointPolicy(
        observation_space,
        action_space,
        effective_ppo_param,
        show_pre_train_debug_msg=True,
        show_progress_bar=True,
        train_frequency=train_frequency_default,
        optimizer_mode=optimizer_mode,
        use_action_masking=str(os.getenv('FLATLAND_USE_ACTION_MASKING', '1')).strip().lower() in ('1', 'true', 'yes', 'on'),
        use_deadlock_avoidance_policy=True
    )
    # Full baseline (MAPPO/PPO-aligned) with references:
    # - PPO:   https://arxiv.org/abs/1707.06347
    # - MAPPO: https://arxiv.org/abs/2103.01955
    # - Impl details: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    # - MAPPO code: https://github.com/marlbenchmark/on-policy
    # - CleanRL PPO code: https://github.com/vwxyzjn/cleanrl
    #
    # Design intent:
    # - Keep PPO update signal dominant (reduce auxiliary over-coupling).
    # - Encourage exploration at decision points without forcing high route prior.
    # - Avoid over-aggressive heuristic/offensive defaults in early curriculum.
    policy.surrogate_eps_clip = float(np.clip(_env_float('FLATLAND_CLIP_EPS', 0.20), 0.05, 0.40))
    policy.weight_entropy = float(np.clip(_env_float('FLATLAND_WEIGHT_ENTROPY', 0.02), 0.0, 1.0))
    policy.reward_scale = float(np.clip(_env_float('FLATLAND_REWARD_SCALE', 0.12), 0.01, 0.25))
    policy.weight_loss = float(np.clip(_env_float('FLATLAND_WEIGHT_VALUE', 1.00), 0.1, 5.0))
    policy.stability_guard_start_episode = 900
    policy.stability_guard_hard_episode = 1800
    policy.ppo_target_kl = 0.020
    policy.ppo_max_kl = 0.050
    policy.ppo_emergency_kl = 0.12
    policy.ppo_emergency_kl_hard = 0.25
    policy.ratio_guard_soft = 1.15
    policy.ratio_guard_soft_low = 0.85
    policy.ratio_guard_hard = 1.22
    policy.ratio_guard_hard_low = 0.75
    policy.max_hard_batches_before_lr_decay = 4
    policy.hard_spike_streak_limit = 4
    policy.actor_lr_min_factor = 0.70
    policy.actor_lr_decay_on_instability = 0.88
    policy.max_eps_random = float(np.clip(_env_float('FLATLAND_MAX_EPS_RANDOM', 0.10), 0.0, 1.0))
    policy.decision_eps_floor = float(np.clip(_env_float('FLATLAND_DECISION_EPS_FLOOR', 0.08), 0.0, 1.0))
    policy.use_decision_eps_floor = str(os.getenv('FLATLAND_USE_DECISION_EPS_FLOOR', '1')).strip().lower() in ('1', 'true', 'yes', 'on')
    # Keep Full-mode defaults unchanged; Core defaults to diversity OFF.
    adiv_default = 0.0 if core_mode_enabled else 0.12
    policy.weight_action_diversity = float(np.clip(_env_float('FLATLAND_WEIGHT_ACTION_DIVERSITY', adiv_default), 0.0, 2.0))
    policy.action_diversity_gate_threshold = float(np.clip(_env_float('FLATLAND_ACTION_DIVERSITY_GATE_THRESHOLD', 0.50), 0.0, 1.0))
    policy.forward_prob_soft_max = float(np.clip(_env_float('FLATLAND_FORWARD_PROB_SOFT_MAX', 0.60), 0.20, 0.80))
    policy.lr_prob_soft_min = float(np.clip(_env_float('FLATLAND_LR_PROB_SOFT_MIN', 0.14), 0.05, 0.40))
    policy.idle_prob_soft_max = float(np.clip(_env_float('FLATLAND_IDLE_PROB_SOFT_MAX', 0.17), 0.0, 0.40))
    policy.idle_logit_penalty = float(np.clip(_env_float('FLATLAND_IDLE_LOGIT_PENALTY', 1.35), 0.0, 4.0))
    policy.stop_logit_penalty = float(np.clip(_env_float('FLATLAND_STOP_LOGIT_PENALTY', 1.20), 0.0, 4.0))
    # Auxiliary deadlock supervision can help sparse/deadlock-heavy MAPPO runs
    # when weighted conservatively (PPO/MAPPO practice):
    # PPO paper:   https://arxiv.org/abs/1707.06347
    # MAPPO paper: https://arxiv.org/abs/2103.01955
    aux_default = 0.0 if core_mode_enabled else 0.04
    aux_w = float(np.clip(_env_float('FLATLAND_WEIGHT_AUX_DEADLOCK', aux_default), 0.0, 1.0))
    for _aux_field in ('weight_aux_deadlock', 'weight_aux_dl'):
        if hasattr(policy, _aux_field):
            setattr(policy, _aux_field, aux_w)
    if hasattr(policy, 'aux_deadlock_pos_weight'):
        policy.aux_deadlock_pos_weight = float(np.clip(_env_float('FLATLAND_AUX_DEADLOCK_POS_WEIGHT', 1.5), 1.0, 8.0))
    if hasattr(policy, 'weight_comm'):
        comm_default = 0.0 if core_mode_enabled else 1.5e-4
        policy.weight_comm = float(np.clip(_env_float('FLATLAND_WEIGHT_COMM', comm_default), 0.0, 1.0))

    # SP-Prior: für 5-agent weniger dominant, damit Policy echte Routing-Entscheidungen lernt
    policy.sp_hint_route_prior_prob = float(np.clip(_env_float('FLATLAND_SP_HINT_ROUTE_PRIOR_PROB', 0.20), 0.0, 1.0))
    policy.sp_hint_logit_bonus = float(np.clip(_env_float('FLATLAND_SP_HINT_LOGIT_BONUS', 0.20), 0.0, 4.0))

    # For single-agent core runs, prioritize fast convergence over exploration.
    if core_mode_enabled and simplified_n == 1:
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
        f"train_frequency={int(policy.train_frequency)}, "
        f"use_decision_eps_floor={bool(policy.use_decision_eps_floor)}, "
        f"core_mode={bool(core_mode_enabled)}, "
        f"weight_entropy={policy.weight_entropy:.3f}, "
        f"clip_eps={policy.surrogate_eps_clip:.3f}, "
        f"weight_value={policy.weight_loss:.3f}, "
        f"weight_aux_deadlock={float(getattr(policy, 'weight_aux_deadlock', 0.0)):.3f}, "
        f"weight_action_diversity={float(getattr(policy, 'weight_action_diversity', 0.0)):.3f}, "
        f"weight_comm={float(getattr(policy, 'weight_comm', 0.0)):.5f}"
    )
    print(
        f"   - route_prior: sp_prob={policy.sp_hint_route_prior_prob:.2f}, "
        f"sp_logit_bonus={policy.sp_hint_logit_bonus:.2f}"
    )
    if use_scalable_simple:
        print('   - scalable_simple_profile: encoder_shared=True, use_spatial_attention=False')
    print('   - profile: BASELINE_STABLE_V1 (PPO/MAPPO reference-aligned)')
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
    # Default (fast + robust): shared encoder, spatial attention enabled
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
    FLATLAND_SIMPLIFIED_MAPPO=0 python marl_attention_temporal.py new --DEBUG # off
  FLATLAND_SIMPLIFIED_MAPPO=1 python marl_attention_temporal.py new --DEBUG # 1 agent
  FLATLAND_SIMPLIFIED_MAPPO=5 python marl_attention_temporal.py new --DEBUG # 5 agents (full mode)
  FLATLAND_SIMPLIFIED_MAPPO=10 python marl_attention_temporal.py new --DEBUG # 10 agents (stress test)
  
ARCHITECTURE CONFIGURATION:
    Default: encoder_shared=True, use_spatial_attention=True
        ✅ Robust with spatial coordination (MAAC-style)
        ⚡ Faster than separate-encoder setup
    
  Recommended for CPU: --encoder-shared
    ✅ Keeps spatial attention (multi-agent learning)
    ⚡ 50% faster (~50% fewer parameters)
    
  Maximum speed: --encoder-shared --no-spatial-attention  
    ⚡ 70% faster overall
    ⚠️  Loses agent-agent attention (temporal-only)
    
    For GPU: defaults are also usually a good starting point

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
        choices=['pure_marl', 'pure_marl_dla', 'marl_dla', 'dla', 'dead_lock_avoidance', 'deadlock_avoidance', 'random'],
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
            "  random               -> RandomPolicy (Baseline)"
        )
    )
    
    parser.add_argument(
        '--DEBUG',
        action='store_true',
        dest='debug',
        help='Enable debug mode for DecisionPointObservation (default: off)'
    )
    
    # ====================================================================
    # ARCHITECTURE OPTIMIZATION FLAGS
    # ====================================================================
    # DEFAULT STRATEGY: encoder_shared=True, use_spatial_attention=True
    # ✅ Robust and scalable via spatial attention
    # ⚡ Shared encoder reduces compute overhead
    # ⚡ Optimize further with --no-spatial-attention for temporal-only setup
    parser.add_argument(
        '--encoder-shared',
        action='store_true',
        dest='encoder_shared',
        default=None,
        help='Share single encoder between actor+critic (~50% faster, -50% params). Default: enabled unless --no-encoder-shared or FLATLAND_ENCODER_SHARED=false.'
    )
    parser.add_argument(
        '--no-encoder-shared',
        action='store_false',
        dest='encoder_shared',
        help='Disable shared encoder and use separate actor/critic encoders.'
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
    if args.encoder_shared is None:
        encoder_shared_cli = str(os.getenv('FLATLAND_ENCODER_SHARED', 'true')).strip().lower() in ('1', 'true', 'yes', 'on')
    else:
        encoder_shared_cli = bool(args.encoder_shared)
    use_spatial_attention_cli = not bool(args.no_spatial_attention)  # Default: True (spatial attention enabled)
    
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
        learning_rate=_env_float('FLATLAND_LR', 1.0e-5),
        discount=_env_float('FLATLAND_DISCOUNT', 0.99),
        gae_lambda=_env_float('FLATLAND_GAE_LAMBDA', 0.95),
        use_gpu=USE_GPU_EFFECTIVE,
        max_episodes_in_training_memory=max(4, _env_int('FLATLAND_TRAIN_MEMORY_EPISODES', default_memory_episodes)),
        k_epochs=max(1, _env_int('FLATLAND_K_EPOCHS', default_k_epochs)),
        batch_fraction=min(1.0, max(0.2, _env_float('FLATLAND_BATCH_FRACTION', default_batch_fraction))),
        max_batches_per_training=max(1, _env_int('FLATLAND_MAX_BATCHES', default_max_batches)),
        temporal_window=TEMPORAL_WINDOW,
        encoder_type=os.getenv('FLATLAND_ENCODER_TYPE', 'lstm'),
        encoder_shared=encoder_shared_cli,
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
    # Honor the configured minimum exploration floor even when --eps is 0.0.
    if do_training and min_eps > eps:
        eps = float(min_eps)

    action_masking_enabled = str(os.getenv('FLATLAND_USE_ACTION_MASKING', '1')).strip().lower() in ('1', 'true', 'yes', 'on')
    temporal_attention_enabled = not _DISABLE_TEMPORAL_ATTENTION

    print(
        f"\n[Config] mode={mode}, eps={eps:.4f}, optimizer_mode={optimizer_mode}, "
        f"encoder_shared={encoder_shared_cli}, use_spatial_attention={use_spatial_attention_cli}, "
        f"policy_mode={policy_mode}, debug={debug_mode}"
    )
    if _DISABLE_TEMPORAL_ATTENTION:
        print("[Config] temporal_attention=OFF (FLATLAND_DISABLE_TEMPORAL_ATTENTION=1, temporal_window=1)")
    else:
        print(f"[Config] temporal_attention=ON (temporal_window={TEMPORAL_WINDOW})")
    if USE_CURRICULUM_PHASES:
        print(f"[Config] curriculum_start_phase_index={start_from_phase} ({CURRICULUM_PHASES[start_from_phase]['name']})")

    print("\n[SETTINGS] (easy overview)")
    print(f"  mode                : {mode}")
    print(f"  policy              : {policy_mode}")
    print(f"  optimizer           : {optimizer_mode.lower()}")
    print(f"  eps / min_eps       : {eps:.4f} / {min_eps:.4f}")
    print(f"  encoder_shared      : {'YES' if encoder_shared_cli else 'NO'}")
    print(f"  action_masking      : {'ON' if action_masking_enabled else 'OFF'}")
    print(f"  spatial_attention   : {'ON' if use_spatial_attention_cli else 'OFF'}")
    print(f"  temporal_attention  : {'ON' if temporal_attention_enabled else 'OFF'}")
    print(f"  temporal_window     : {TEMPORAL_WINDOW}")
    print(f"  device              : {'GPU' if USE_GPU_EFFECTIVE else 'CPU'}")

    environment = RailEnvironmentPersistable(
        obs_builder_object_creator=lambda: create_temporal_obs_builder_object(
            debug=debug_mode,
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

    # Use the actual base-obs size so the policy gets a matching state_size.
    _obs_builder_for_size = create_temporal_obs_builder_object(
        debug=debug_mode,
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
            
