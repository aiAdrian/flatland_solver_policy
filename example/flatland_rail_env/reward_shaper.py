# =============================================================================
# reward_shaper.py
# -----------------------------------------------------------------------------
# Reward shapers for Flatland MARL training.
# References:
#   - Ng, Harada, Russell (1999): "Policy Invariance Under Reward Transformations"
#   - Laurent et al. (2021), Flatland Competition (arXiv:2103.16511)
#   - Devlin & Kudenko (2012): "Dynamic potential-based reward shaping"
#   - Ross et al. (2011): "DAgger: A Reduction of Imitation Learning..."
#
# Two shapers:
#   - SimpleRewardShaper:   classic additive shaping (DLA-free)
#   - DLAImitationReward:   pure DLA-imitation reward (decision-points only)
# =============================================================================

from typing import Dict, Optional, Any

import numpy as np
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_argmax, fast_count_nonzero
from flatland.envs.step_utils.states import TrainState

from environment.environment import Environment
from marl_attention_temporal_observation.decision_point_utils import DecisionPointUtils
from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.deadlock_avoidance_policy \
    import DeadLockAvoidancePolicy


# =============================================================================
# SimpleRewardShaper — DLA-free additive shaping
# =============================================================================
class SimpleRewardShaper:
    """Additive reward shaper — DLA-free.

    Per agent per step:
        + done_bonus * done_decay(t)  when agent reaches target (once)
        + all_done_bonus              when ALL agents are done (once per agent)
        + progress_bonus * Δsp        per cell of shortest-path progress
        - step_penalty                while active
        - useless_stop_penalty        when STOP but forward is free
        - deadlock_penalty            while in local deadlock
        - final_not_solved x5         one-shot at last step if not done
    """

    ACTION_STOP = 4

    def __init__(
        self,
        step_penalty: float = 0.0,
        done_bonus: float = 5.0,
        all_done_bonus: float = 10.0,
        deadlock_penalty: float = 1.0,
        final_not_solved_penalty: float = 1.0,
        useless_stop_penalty: float = 0.0,
        progress_bonus: float = 0.0,
        done_decay_steps: int = 0,
        done_bonus_floor: float = 1.0,
        **_unused_legacy_kwargs,
    ):
        self.step_penalty = float(step_penalty)
        self.done_bonus = float(done_bonus)
        self.all_done_bonus = float(all_done_bonus)
        self.deadlock_penalty = float(deadlock_penalty)
        self.final_not_solved_penalty = float(final_not_solved_penalty)
        self.useless_stop_penalty = float(useless_stop_penalty)
        self.progress_bonus = float(progress_bonus)
        self.done_decay_steps = int(done_decay_steps)
        self.done_bonus_floor = float(done_bonus_floor)

        self._last_episode_step = -1
        self._rewarded_done: Dict[int, bool] = {}
        self._all_done_given: bool = False
        self._final_penalty_given: bool = False
        self._episode_deadlocks = set()
        self._last_sp_distance: Dict[int, float] = {}
        self._diag_done_bonus_sum = 0.0

    def _reset_episode(self, env: Environment):
        n = len(env.raw_env.agents)
        self._rewarded_done = {i: False for i in range(n)}
        self._all_done_given = False
        self._final_penalty_given = False
        self._episode_deadlocks = set()
        self._last_sp_distance = {}
        self._diag_done_bonus_sum = 0.0

    @staticmethod
    def _build_agent_map(env: Environment) -> np.ndarray:
        raw = env.raw_env
        agent_map = np.full((raw.height, raw.width), -1, dtype=np.int32)
        for a in raw.agents:
            if a.position is not None:
                agent_map[a.position] = int(a.handle)
        return agent_map

    def _done_decay(self, raw_env) -> float:
        if self.done_decay_steps <= 0:
            return 1.0
        elapsed = float(raw_env._elapsed_steps)
        return max(0.0, 1.0 - elapsed / float(self.done_decay_steps))

    @staticmethod
    def _is_useless_stop(raw_env, agent, agent_map: np.ndarray, action: Optional[int]) -> bool:
        if agent.position is None or agent.state == TrainState.DONE:
            return False
        if action is None or int(action) != SimpleRewardShaper.ACTION_STOP:
            return False
        transitions = raw_env.rail.get_transitions(
            agent.position[0], agent.position[1], agent.direction
        )
        n_trans = fast_count_nonzero(transitions)
        if n_trans == 0:
            return False
        if n_trans == 1:
            fwd_dir = int(fast_argmax(transitions))
            target = get_new_position(agent.position, fwd_dir)
            if (target[0] < 0 or target[0] >= raw_env.height or
                    target[1] < 0 or target[1] >= raw_env.width):
                return False
            return bool(agent_map[target[0], target[1]] == -1)
        else:
            for nd in range(4):
                if not transitions[nd]:
                    continue
                target = get_new_position(agent.position, nd)
                if (target[0] < 0 or target[0] >= raw_env.height or
                        target[1] < 0 or target[1] >= raw_env.width):
                    return False
                if agent_map[target[0], target[1]] != -1:
                    return False
            return True

    def _sp_distance(self, env: Environment, handle: int) -> Optional[float]:
        raw = env.raw_env
        agent = raw.agents[handle]
        if agent.position is None or agent.direction is None:
            return None
        try:
            dm = raw.distance_map.get()
            d = float(dm[handle, agent.position[0], agent.position[1], int(agent.direction)])
            return d if np.isfinite(d) else None
        except Exception:
            return None

    def get_diagnostics(self) -> dict:
        return {
            "done_bonus_sum": self._diag_done_bonus_sum,
            "deadlocks": len(self._episode_deadlocks),
        }

    def __call__(self, reward, terminal, info, env: Environment, actions=None):
        raw = env.raw_env
        cur_step = int(raw._elapsed_steps)

        if (self._last_episode_step < 0 or
                cur_step <= 1 or
                cur_step < self._last_episode_step):
            self._reset_episode(env)
        self._last_episode_step = cur_step

        shaped = dict(reward)
        agent_map = self._build_agent_map(env)
        all_done = all(a.state == TrainState.DONE for a in raw.agents)

        is_final_step = (cur_step >= raw._max_episode_steps - 1)
        give_final_penalty = (
            is_final_step and not all_done and not self._final_penalty_given
        )

        active_handles = [
            int(a.handle) for a in raw.agents
            if a.state != TrainState.DONE and a.position is not None
        ]
        deadlock_check = len(active_handles) > 1
        give_team = all_done and not self._all_done_given

        done_decay = self._done_decay(raw)
        done_bonus_factor = (
            self.done_bonus_floor + (1.0 - self.done_bonus_floor) * done_decay
            if self.done_decay_steps > 0
            else 1.0
        )

        for handle in env.get_agent_handles():
            agent = raw.agents[handle]
            r = 0.0

            if agent.state > TrainState.WAITING:
                if agent.state < TrainState.DONE:
                    r -= self.step_penalty

                    if (deadlock_check and agent.position is not None
                        and DecisionPointUtils.is_local_deadlock(raw, agent, agent_map)):
                        self._episode_deadlocks.add(handle)
                        r -= self.deadlock_penalty

                    if self.useless_stop_penalty > 0.0 and actions is not None:
                        action = actions.get(handle, None) if isinstance(actions, dict) else None
                        if self._is_useless_stop(raw, agent, agent_map, action):
                            r -= self.useless_stop_penalty

                    if self.progress_bonus > 0.0:
                        cur_d = self._sp_distance(env, handle)
                        if cur_d is not None:
                            last_d = self._last_sp_distance.get(handle, None)
                            if last_d is not None and cur_d < last_d:
                                r += self.progress_bonus * float(last_d - cur_d)
                            self._last_sp_distance[handle] = cur_d

                if agent.state == TrainState.DONE and not self._rewarded_done[handle]:
                    bonus = self.done_bonus * done_bonus_factor
                    r += bonus
                    self._diag_done_bonus_sum += bonus
                    self._rewarded_done[handle] = True

                if give_team:
                    r += self.all_done_bonus

                if give_final_penalty and agent.state < TrainState.DONE:
                    r -= self.final_not_solved_penalty * 5.0

            shaped[handle] = float(r)

        if give_team:
            self._all_done_given = True
        if give_final_penalty:
            self._final_penalty_given = True

        return shaped


# =============================================================================
# DLAImitationReward — pure DLA-imitation (decision points only)
# -----------------------------------------------------------------------------
# Per step (decision points only — SWITCH/MERGING/PRE_M):
#   +match_bonus       if agent_action == DLA_action
#   +deviation_penalty if agent_action != DLA_action
#   else 0             (no signal at FORWARD_ONLY cells)
#
# Terminal:
#   -(max_episode_steps + fail_extra_offset) per agent NOT done
#   +all_done_bonus per agent if ALL done
#
# Final reward × reward_scale (PPO value-loss stability)
# =============================================================================

class DLAImitationReward:
    """Clean DLA-imitation reward for Flatland MARL.

    Args:
        match_bonus:        reward for matching DLA action at decision point
        deviation_penalty:  penalty for deviating from DLA at decision point
        all_done_bonus:     bonus per agent if ALL agents done
        fail_extra_offset:  per-agent failure penalty = -(max_steps + offset)
        reward_scale:       uniform scale on shaped output (default 0.01)
    """

    def __init__(
        self,
        match_bonus: float = 1.0,
        deviation_penalty: float = -1.0,
        all_done_bonus: float = 10.0,
        fail_extra_offset: float = 10.0,
        reward_scale: float = 0.01,
    ):
        self.match_bonus = float(match_bonus)
        self.deviation_penalty = float(deviation_penalty)
        self.all_done_bonus = float(all_done_bonus)
        self.fail_extra_offset = float(fail_extra_offset)
        self.reward_scale = float(reward_scale)

        self._dla: Optional[DeadLockAvoidancePolicy] = None
        self._terminal_given = False
        self._last_episode_step = -1

        self._diag = {
            "matches": 0,
            "deviations": 0,
            "fail_penalties_total": 0.0,
            "all_done_bonus_total": 0.0,
            "episodes_finished": 0,
            "episodes_all_done": 0,
        }

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _is_decision_point(raw_env, agent) -> bool:
        """Return True if agent is at a SWITCH / MERGING / PRE_M cell."""
        if agent.position is None or agent.state == TrainState.DONE:
            return False
        try:
            cell_type = DecisionPointUtils.classify_cell_type(agent, raw_env)
            return cell_type in ("SWITCH", "MERGING", "PRE_M")
        except Exception:
            return False

    def _reset_episode(self, env: Environment):
        self._terminal_given = False
        try:
            if self._dla is None:
                self._dla = DeadLockAvoidancePolicy(
                    env.raw_env, action_size=5, enable_eps=False
                )
            self._dla.reset(env)
            try:
                self._dla.start_step(False)
            except Exception:
                pass
        except Exception as e:
            print(f"[DLAImitationReward] DLA init failed: {e}")
            self._dla = None

    def _get_dla_action(self, handle: int) -> Optional[int]:
        if self._dla is None:
            return None
        try:
            return int(self._dla.act(handle, None, eps=0.0))
        except Exception:
            return None

    # ----------------------------------------------------------------- main

    def __call__(self, reward, terminal, info, env: Environment, actions=None):
        """Compute shaped reward dict (REPLACES env reward)."""
        raw = env.raw_env
        cur_step = int(raw._elapsed_steps)

        # Episode reset detection
        if (self._last_episode_step < 0 or
                cur_step <= 1 or
                cur_step < self._last_episode_step):
            self._reset_episode(env)
        self._last_episode_step = cur_step

        # Refresh DLA's per-step state (precomputed paths)
        if self._dla is not None:
            try:
                self._dla.start_step(False)
            except Exception:
                pass

        # Start with all zeros (replace env reward entirely)
        shaped: Dict[int, float] = {h: 0.0 for h in reward.keys()}

        # ─── PER-STEP: DLA agreement at decision points only ──────────
        if actions is not None:
            for handle, agent_action in actions.items():
                if not isinstance(handle, int):
                    continue
                try:
                    agent = raw.agents[handle]
                except (IndexError, KeyError):
                    continue
                if agent.state == TrainState.DONE:
                    continue

                # Filter: only decision points (SWITCH/MERGING/PRE_M).
                # FORWARD_ONLY cells: DLA action is trivial → no signal.
                if not self._is_decision_point(raw, agent):
                    continue

                dla_action = self._get_dla_action(handle)
                if dla_action is None:
                    continue

                if int(agent_action) == int(dla_action):
                    shaped[handle] = shaped.get(handle, 0.0) + self.match_bonus
                    self._diag["matches"] += 1
                else:
                    shaped[handle] = shaped.get(handle, 0.0) + self.deviation_penalty
                    self._diag["deviations"] += 1

        # ─── TERMINAL: success / failure ──────────────────────────────
        all_term = (
            terminal.get("__all__", False)
            if isinstance(terminal, dict)
            else bool(terminal)
        )
        is_final_step = (cur_step >= raw._max_episode_steps - 1)
        episode_ending = (all_term or is_final_step) and not self._terminal_given

        if episode_ending:
            self._terminal_given = True
            self._diag["episodes_finished"] += 1

            max_steps = int(raw._max_episode_steps)
            n_agents = len(raw.agents)
            done_count = sum(
                1 for ag in raw.agents if ag.state == TrainState.DONE
            )

            # Per-agent failure penalty
            fail_penalty = -(max_steps + self.fail_extra_offset)
            for ag in raw.agents:
                if ag.state != TrainState.DONE:
                    h = int(ag.handle)
                    shaped[h] = shaped.get(h, 0.0) + fail_penalty
                    self._diag["fail_penalties_total"] += fail_penalty

            # All-done bonus
            if done_count == n_agents:
                self._diag["episodes_all_done"] += 1
                for ag in raw.agents:
                    h = int(ag.handle)
                    shaped[h] = shaped.get(h, 0.0) + self.all_done_bonus
                    self._diag["all_done_bonus_total"] += self.all_done_bonus

        # Scale rewards uniformly (PPO value-loss stability)
        if self.reward_scale != 1.0:
            for h in shaped:
                shaped[h] *= self.reward_scale

        return shaped

    # ----------------------------------------------------------------- diag

    def get_diagnostics(self) -> dict:
        d = dict(self._diag)
        total = d["matches"] + d["deviations"]
        d["match_rate"] = d["matches"] / total if total > 0 else 0.0
        return d


# =============================================================================
# OutcomeBasedReward — outcome-driven, deadlock-aware, mild DLA hint
# -----------------------------------------------------------------------------
# Philosophy:
#   "We want agents to LEARN better than DLA — not imitate it."
#
#   DLA is greedy on shortest-path → wastes time at MERGING (60% Stop).
#   A trained policy can learn:
#     - take a longer alternative path (avoids waiting)
#     - manage merge-conflicts proactively
#     - cooperate as a team
#
# Reward components (sparse + dense outcome-driven):
#
#   Per step (every agent, every step while active):
#     -step_penalty                    time pressure
#
#   Per step in local deadlock:
#     -deadlock_penalty                cost of being stuck (continuous)
#
#   Per agent done (one-shot):
#     +done_bonus + (max_steps - elapsed) * time_saved_factor
#                                       reward EARLY arrival, not just arrival
#
#   Per terminal step (one-shot):
#     +all_done_bonus  (only if ALL done)   teamwork bonus
#     -fail_penalty per agent NOT done       per-agent failure
#
#   Optional soft DLA hint at decision points (very small):
#     +match_bonus                      slight encouragement to consider DLA
#                                       (NO deviation penalty — abweichen ist OK!)
#
# All values × reward_scale at the end (PPO value-loss stability).
#
#
# 
# Per step (alle agents, alle steps):
#  -1                                     time pressure (immer)
#
# Per agent terminal (DONE):
#  +max_steps                             "you saved (max - actual) steps!"
#  Dies belohnt FRÜH ankommen, nicht nur ankommen
#
# Per agent in deadlock (lokal erkannt):
#  -deadlock_penalty                      "kostet pro step im deadlock"
#
# Episode end:
#   +all_done_bonus  (wenn alle DONE)      teamwork
#   -fail_penalty per agent NOT done      
#
# Optional: DLA-soft-anchor (mild)
#   +small_match_bonus an decision points  "DLA-Tipps respektieren"
#   KEIN deviation_penalty                 "Abweichen ist OK!"
# =============================================================================

class OutcomeBasedReward:
    """Outcome-driven reward — encourages learning *better* than DLA.

    Args:
        step_penalty:           per-step cost (time pressure). default 1.0
        deadlock_penalty:       per-step cost while in local deadlock. default 5.0
        done_bonus:             flat reward when an agent reaches target. default 50.0
        time_saved_factor:      bonus per step saved on done (max_steps - elapsed).
                                default 1.0  (so total bonus ~ done_bonus + steps_saved)
        all_done_bonus:         per-agent bonus if ALL done at episode end. default 100.0
        fail_penalty:           per-agent penalty if not done at episode end. default 200.0
        match_bonus:            tiny bonus when action == DLA action at decision point.
                                default 0.0  (off — pure outcome-based)
                                set to 0.1-0.5 for soft DLA-anchor.
        reward_scale:           uniform scaler (default 0.01 for PPO stability)
    """

    def __init__(
        self,
        step_penalty: float = 1.0,
        deadlock_penalty: float = 5.0,
        done_bonus: float = 50.0,
        time_saved_factor: float = 1.0,
        all_done_bonus: float = 100.0,
        fail_penalty: float = 200.0,
        match_bonus: float = 0.0,
        reward_scale: float = 0.01,
    ):
        self.step_penalty = float(step_penalty)
        self.deadlock_penalty = float(deadlock_penalty)
        self.done_bonus = float(done_bonus)
        self.time_saved_factor = float(time_saved_factor)
        self.all_done_bonus = float(all_done_bonus)
        self.fail_penalty = float(fail_penalty)
        self.match_bonus = float(match_bonus)
        self.reward_scale = float(reward_scale)

        self._dla: Optional[DeadLockAvoidancePolicy] = None
        self._terminal_given = False
        self._last_episode_step = -1
        self._rewarded_done: Dict[int, bool] = {}

        # Diagnostics
        self._diag = {
            "step_pen_total": 0.0,
            "deadlock_pen_total": 0.0,
            "done_bonus_total": 0.0,
            "time_saved_total": 0.0,
            "match_bonus_total": 0.0,
            "fail_pen_total": 0.0,
            "all_done_bonus_total": 0.0,
            "deadlocks": 0,
            "matches": 0,
        }

    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _is_decision_point(raw_env, agent) -> bool:
        if agent.position is None or agent.state == TrainState.DONE:
            return False
        try:
            cell_type = DecisionPointUtils.classify_cell_type(agent, raw_env)
            return cell_type in ("SWITCH", "MERGING", "PRE_M")
        except Exception:
            return False

    def _reset_episode(self, env: Environment):
        n = len(env.raw_env.agents)
        self._terminal_given = False
        self._rewarded_done = {i: False for i in range(n)}
        self._diag = {k: 0 for k in self._diag}

        # Init DLA only if we use match_bonus
        if self.match_bonus > 0.0:
            try:
                if self._dla is None:
                    self._dla = DeadLockAvoidancePolicy(
                        env.raw_env, action_size=5, enable_eps=False
                    )
                self._dla.reset(env)
                self._dla.start_step(False)
            except Exception as e:
                print(f"[OutcomeBasedReward] DLA init failed: {e}")
                self._dla = None
        else:
            self._dla = None

    @staticmethod
    def _build_agent_map(raw_env) -> np.ndarray:
        agent_map = np.full((raw_env.height, raw_env.width), -1, dtype=np.int32)
        for a in raw_env.agents:
            if a.position is not None:
                agent_map[a.position] = int(a.handle)
        return agent_map

    def _get_dla_action(self, handle: int) -> Optional[int]:
        if self._dla is None:
            return None
        try:
            return int(self._dla.act(handle, None, eps=0.0))
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────────────
    def __call__(self, reward, terminal, info, env: Environment, actions=None):
        raw = env.raw_env
        cur_step = int(raw._elapsed_steps)
        max_steps = int(raw._max_episode_steps)

        # Episode reset detection
        if (self._last_episode_step < 0 or
                cur_step <= 1 or
                cur_step < self._last_episode_step):
            self._reset_episode(env)
        self._last_episode_step = cur_step

        # Refresh DLA per-step state
        if self._dla is not None:
            try:
                self._dla.start_step(False)
            except Exception:
                pass

        # Start with all zeros (replace env reward)
        shaped: Dict[int, float] = {h: 0.0 for h in reward.keys()}
        agent_map = self._build_agent_map(raw)

        # ─── PER STEP rewards ──────────────────────────────────────
        for handle in env.get_agent_handles():
            try:
                agent = raw.agents[handle]
            except (IndexError, KeyError):
                continue

            # Already done agents: no further per-step costs
            if agent.state == TrainState.DONE:
                # First-time DONE bonus
                if not self._rewarded_done.get(handle, False):
                    elapsed = cur_step
                    time_saved = max(0, max_steps - elapsed)
                    bonus = self.done_bonus + self.time_saved_factor * time_saved
                    shaped[handle] += bonus
                    self._diag["done_bonus_total"] += self.done_bonus
                    self._diag["time_saved_total"] += self.time_saved_factor * time_saved
                    self._rewarded_done[handle] = True
                continue

            # Inactive (waiting/off-map): no penalty either
            if agent.state <= TrainState.WAITING:
                continue

            # ── Time pressure ──
            shaped[handle] -= self.step_penalty
            self._diag["step_pen_total"] += self.step_penalty

            # ── Deadlock penalty ──
            if (agent.position is not None and
                    DecisionPointUtils.is_local_deadlock(raw, agent, agent_map)):
                shaped[handle] -= self.deadlock_penalty
                self._diag["deadlock_pen_total"] += self.deadlock_penalty
                self._diag["deadlocks"] += 1

            # ── Optional soft DLA hint at decision points ──
            if (self.match_bonus > 0.0 and actions is not None
                    and self._is_decision_point(raw, agent)):
                action_taken = actions.get(handle, None)
                if action_taken is not None:
                    dla_action = self._get_dla_action(handle)
                    if dla_action is not None and int(action_taken) == int(dla_action):
                        shaped[handle] += self.match_bonus
                        self._diag["match_bonus_total"] += self.match_bonus
                        self._diag["matches"] += 1

        # ─── TERMINAL: success / failure ──────────────────────────
        all_term = (
            terminal.get("__all__", False)
            if isinstance(terminal, dict)
            else bool(terminal)
        )
        is_final_step = (cur_step >= max_steps - 1)
        episode_ending = (all_term or is_final_step) and not self._terminal_given

        if episode_ending:
            self._terminal_given = True
            n_agents = len(raw.agents)
            done_count = sum(1 for a in raw.agents if a.state == TrainState.DONE)

            # Per-agent failure
            for ag in raw.agents:
                if ag.state != TrainState.DONE:
                    h = int(ag.handle)
                    shaped[h] -= self.fail_penalty
                    self._diag["fail_pen_total"] += self.fail_penalty

            # Team bonus
            if done_count == n_agents:
                for ag in raw.agents:
                    h = int(ag.handle)
                    shaped[h] += self.all_done_bonus
                    self._diag["all_done_bonus_total"] += self.all_done_bonus

        # ─── Apply uniform reward scale ──────────────────────────
        if self.reward_scale != 1.0:
            for h in shaped:
                shaped[h] *= self.reward_scale

        return shaped

    # ─────────────────────────────────────────────────────────────────────
    def get_diagnostics(self) -> dict:
        d = dict(self._diag)
        d["net_per_step"] = -d["step_pen_total"] - d["deadlock_pen_total"]
        d["net_terminal"] = (
            d["done_bonus_total"] + d["time_saved_total"]
            + d["all_done_bonus_total"] - d["fail_pen_total"]
        )
        return d

# =============================================================================
# CounterfactualCTDEReward — counterfactual reward gegen DLA-Baseline (CTDE)
# -----------------------------------------------------------------------------
# Per-agent reward AT EPISODE END:
#   r(i) = (done_i ? +1 : -1)                          # ±1 base
#        + done_i * (DLA_steps_i - MARL_steps_i)/DLA_steps_i  # time bonus
#        - (done_DLA_count - done_MARL_count)          # global counterfactual
#        + all_done_MARL * all_done_bonus              # team bonus
#
# Per-step reward: 0 (or tiny step_penalty for dense signal)
#
# Requires DLA-baseline cache (precomputed via precompute_dla_baseline()).
# =============================================================================

class CounterfactualCTDEReward:
    """Counterfactual reward — encourages MARL to BEAT DLA.

    Args:
        all_done_bonus:   Bonus when ALL MARL agents reach goal. default +2.0
        step_penalty:     Tiny per-step penalty (dense signal). default 0.0
        reward_scale:     Uniform scaler for PPO stability. default 1.0
        dla_cache:        Pre-computed dict {fingerprint → {done_set, steps_per_agent}}
        verbose:          Log per-episode stats. default False
    """

    def __init__(
        self,
        all_done_bonus: float = 2.0,
        step_penalty: float = 0.0,
        reward_scale: float = 1.0,
        dla_cache: Optional[Dict] = None,
        verbose: bool = False,
    ):
        self.all_done_bonus = float(all_done_bonus)
        self.step_penalty = float(step_penalty)
        self.reward_scale = float(reward_scale)
        self.verbose = verbose
        
        # DLA cache: fingerprint → {'done_set': set, 'steps_per_agent': dict}
        self._dla_cache: Dict = dla_cache if dla_cache is not None else {}
        self._dla: Optional[DeadLockAvoidancePolicy] = None  # for live-shadow fallback
        
        # Per-episode state (reset every episode)
        self._terminal_given = False
        self._last_episode_step = -1
        self._marl_steps_to_done: Dict[int, int] = {}  # handle → step_when_done
        self._was_done: Dict[int, bool] = {}            # handle → already-done flag
        self._cached_dla_data: Optional[Dict] = None    # current episode's DLA-baseline
        
        # Aggregate diagnostics
        self._diag = {
            "n_marl_better": 0,    # MARL done_count > DLA
            "n_marl_worse":  0,    # MARL done_count < DLA
            "n_tied":        0,    # equal done_count
            "n_all_done":    0,    # all MARL agents reached goal
            "n_episodes":    0,
            "n_cache_hits":  0,
            "n_cache_misses": 0,   # forced live-rollout (slow)
            "sum_reward":    0.0,
            "sum_done_marl": 0,
            "sum_done_dla":  0,
        }

    # ─────────────────────────────────────────────────────────
    @staticmethod
    def env_fingerprint(raw_env) -> str:
        """Compute deterministic, process-stable hash for env identity.

        Uses SHA1 over a stable string representation. The returned hex
        digest is the same in every Python process — unlike Python's
        builtin hash() which is randomized per-process.

        We hash the agent layout (start + target + direction), which is
        unique per cached env. We do NOT use raw_env.random_seed because
        it's the constructor seed (typically 42 for ALL cached envs).
        """
        import hashlib
        try:
            parts = [
                f"{raw_env.width}x{raw_env.height}",
                f"n{len(raw_env.agents)}",
            ]
            for a in raw_env.agents:
                parts.append(
                    f"{int(a.initial_position[0])},{int(a.initial_position[1])}->"
                    f"{int(a.target[0])},{int(a.target[1])}:"
                    f"{int(a.initial_direction)}"
                )
            key = "|".join(parts)
            return hashlib.sha1(key.encode("utf-8")).hexdigest()
        except Exception:
            return "unknown"

    # ─────────────────────────────────────────────────────────
    def _reset_episode(self, env: Environment):
        raw = env.raw_env
        n = len(raw.agents)
        
        self._terminal_given = False
        self._marl_steps_to_done = {}
        self._was_done = {i: False for i in range(n)}
        
        # Lookup DLA baseline for this env
        fp = self.env_fingerprint(raw)
        
        if fp in self._dla_cache:
            self._cached_dla_data = self._dla_cache[fp]
            self._diag["n_cache_hits"] += 1
        else:
            self._diag["n_cache_misses"] += 1
            if self.verbose:
                print(f"[CFReward] CACHE MISS for fingerprint {fp} — using fallback "
                      f"(DLA_steps=max_steps for all agents)")
            # Fallback: assume DLA failed everywhere
            max_steps = int(raw._max_episode_steps)
            self._cached_dla_data = {
                'done_set': set(),
                'steps_per_agent': {a.handle: max_steps for a in raw.agents},
            }

    # ─────────────────────────────────────────────────────────
    def __call__(self, reward, terminal, info, env: Environment, actions=None):
        """Compute shaped reward dict for this step. Replaces env reward."""
        raw = env.raw_env
        cur_step = int(raw._elapsed_steps)
        max_steps = int(raw._max_episode_steps)

        # Episode reset detection (same logic as other shapers)
        if (self._last_episode_step < 0 or
                cur_step <= 1 or
                cur_step < self._last_episode_step):
            self._reset_episode(env)
        self._last_episode_step = cur_step

        # Initialize zero rewards
        shaped: Dict[int, float] = {h: 0.0 for h in reward.keys()}

        # ─── PER STEP: track when each MARL agent reaches goal ───────
        for handle in env.get_agent_handles():
            try:
                agent = raw.agents[handle]
            except (IndexError, KeyError):
                continue
            
            # Active agent: small step penalty (optional)
            if (agent.state > TrainState.WAITING and
                    agent.state != TrainState.DONE):
                shaped[handle] -= self.step_penalty
            
            # First-time DONE: record step
            if agent.state == TrainState.DONE and not self._was_done.get(handle, False):
                self._marl_steps_to_done[handle] = cur_step
                self._was_done[handle] = True

        # ─── TERMINAL: Counterfactual reward ─────────────────────────
        all_term = (
            terminal.get("__all__", False)
            if isinstance(terminal, dict)
            else bool(terminal)
        )
        is_final_step = (cur_step >= max_steps - 1)
        episode_ending = (all_term or is_final_step) and not self._terminal_given

        if episode_ending:
            self._terminal_given = True
            self._apply_counterfactual_reward(shaped, raw, max_steps)

        # ─── Apply scale ──
        if self.reward_scale != 1.0:
            for h in shaped:
                shaped[h] *= self.reward_scale

        return shaped

    # ─────────────────────────────────────────────────────────
    def _apply_counterfactual_reward(self, shaped: Dict, raw_env, max_steps: int):
        """Compute terminal counterfactual reward, mutates `shaped` dict."""
        n_agents = len(raw_env.agents)
        
        # MARL final state
        marl_done_set = {h for h, was in self._was_done.items() if was}
        marl_done_count = len(marl_done_set)
        all_done_marl = (marl_done_count == n_agents)
        
        # DLA baseline
        dla_done_set = self._cached_dla_data['done_set']
        dla_steps_dict = self._cached_dla_data['steps_per_agent']
        dla_done_count = len(dla_done_set)
        
        # Global done diff (same for all agents — CTDE!)
        done_diff = dla_done_count - marl_done_count  # positive = MARL worse
        cf_term = -done_diff
        
        # All done bonus (same for all agents)
        team_bonus = self.all_done_bonus if all_done_marl else 0.0
        
        # Per-agent reward
        for agent in raw_env.agents:
            handle = int(agent.handle)
            done_marl = (handle in marl_done_set)
            
            # T1: base ±1
            base = +1.0 if done_marl else -1.0
            
            # T2: time bonus (only if MARL done)
            # T2: time bonus (only if MARL done)
            time_bonus = 0.0
            if done_marl:
                marl_st = self._marl_steps_to_done.get(handle, max_steps)
                # If DLA failed for this agent → use max_steps (favorable to MARL)
                dla_st = float(dla_steps_dict.get(handle, max_steps))
                time_bonus = (dla_st - marl_st) / max(dla_st, 1.0)
                # Clamp to [-1, +1] for stability
                time_bonus = max(-1.0, min(1.0, time_bonus))
            
            # T3: counterfactual diff (already computed, same for all)
            # T4: team bonus (already computed, same for all)
            
            r = base + time_bonus + cf_term + team_bonus
            shaped[handle] = shaped.get(handle, 0.0) + r
        
        # Aggregate diagnostics
        self._diag["n_episodes"] += 1
        self._diag["sum_done_marl"] += marl_done_count
        self._diag["sum_done_dla"] += dla_done_count
        if marl_done_count > dla_done_count:
            self._diag["n_marl_better"] += 1
        elif marl_done_count < dla_done_count:
            self._diag["n_marl_worse"] += 1
        else:
            self._diag["n_tied"] += 1
        if all_done_marl:
            self._diag["n_all_done"] += 1
        
        # Track avg reward for diag
        avg_r = sum(
            shaped.get(int(a.handle), 0.0) for a in raw_env.agents
        ) / max(n_agents, 1)
        self._diag["sum_reward"] += avg_r
        
        if self.verbose:
            print(f"[CFReward] ep={self._diag['n_episodes']}  "
                  f"MARL_done={marl_done_count}/{n_agents}  "
                  f"DLA_done={dla_done_count}/{n_agents}  "
                  f"all_done_MARL={all_done_marl}  "
                  f"avg_reward={avg_r:+.2f}")
    
    # ─────────────────────────────────────────────────────────
    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic stats for logging."""
        n = max(self._diag["n_episodes"], 1)
        cache_total = max(self._diag["n_cache_hits"] + self._diag["n_cache_misses"], 1)
        return {
            "marl_better_pct":  self._diag["n_marl_better"] / n,
            "marl_worse_pct":   self._diag["n_marl_worse"] / n,
            "tied_pct":         self._diag["n_tied"] / n,
            "all_done_pct":     self._diag["n_all_done"] / n,
            "avg_done_marl":    self._diag["sum_done_marl"] / n,
            "avg_done_dla":     self._diag["sum_done_dla"] / n,
            "avg_reward":       self._diag["sum_reward"] / n,
            "cache_hit_rate":   self._diag["n_cache_hits"] / cache_total,
            "n_episodes":       self._diag["n_episodes"],
        }
    
    def reset_diagnostics(self):
        for k in self._diag:
            self._diag[k] = 0
        self._diag["sum_reward"] = 0.0


# =============================================================================
# DLA Baseline Pre-compute (run once before CounterfactualCTDEReward)
# =============================================================================
def precompute_dla_baseline(
    env: Environment,
    cache_path: str = "dla_baseline_cache.pkl",
    n_envs: Optional[int] = None,
    verbose: bool = True,
) -> Dict[int, Dict]:
    """Run DLA on every cached env, store (done_set, steps_per_agent) keyed by fingerprint.
    
    Args:
        env:        Loaded Flatland environment (with cached envs).
        cache_path: Where to save/load the cache.
        n_envs:     If set, only process first N envs (debug).
        verbose:    Progress logging.
    
    Returns:
        dict: {fingerprint: {'done_set': set, 'steps_per_agent': {handle: int}}}
    """
    import os
    import pickle
    import time
    
    if os.path.exists(cache_path):
        if verbose:
            print(f"[DLA-Baseline] Loading existing cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        if verbose:
            print(f"[DLA-Baseline] Loaded {len(cache)} cached entries.")
        return cache
    
    cache: Dict[int, Dict] = {}
    
    # Access loaded env-pool
    loaded_envs = getattr(env, '_loaded_env', None)
    if loaded_envs is None or len(loaded_envs) == 0:
        raise RuntimeError("[DLA-Baseline] env._loaded_env is empty. Call build_environment() first.")
    
    total = len(loaded_envs) if n_envs is None else min(n_envs, len(loaded_envs))
    if verbose:
        print(f"[DLA-Baseline] Pre-computing DLA rollouts for {total} envs...")
    
    t0 = time.perf_counter()
    
    dla = DeadLockAvoidancePolicy(raw, action_size=5, enable_eps=False)
    for env_idx in range(total):
        # Reset env to specific cached env
        env._loaded_env_itr = env_idx
        obs, info = env.reset()
        raw = env.raw_env
        
        fp = CounterfactualCTDEReward.env_fingerprint(raw)
        if fp in cache:
            continue  # already done (e.g., duplicate seeds)
        
        # Initialize DLA for this env
        dla.reset(env)
        dla.start_step(False)
        
        max_steps = int(raw._max_episode_steps)
        steps_per_agent: Dict[int, int] = {}
        done_set = set()
        
        # Rollout DLA
        for step_i in range(max_steps):
            dla.start_step(False)
            actions = {}
            for h in env.get_agent_handles():
                try:
                    actions[h] = int(dla.act(h, None, eps=0.0))
                except Exception:
                    actions[h] = 0  # safe fallback
            
            obs, rewards, dones, info = env.step(actions)
            
            # Record newly-done agents
            for h, ag in enumerate(raw.agents):
                if ag.state == TrainState.DONE and h not in done_set:
                    done_set.add(h)
                    steps_per_agent[h] = step_i + 1
            
            if dones.get("__all__", False):
                break
        
        # Fill in steps for not-done agents (use max_steps)
        for ag in raw.agents:
            h = int(ag.handle)
            if h not in steps_per_agent:
                steps_per_agent[h] = max_steps
        
        cache[fp] = {
            'done_set': set(done_set),
            'steps_per_agent': dict(steps_per_agent),
            'n_agents': len(raw.agents),
        }
        
        if verbose and (env_idx + 1) % 25 == 0:
            elapsed = time.perf_counter() - t0
            done_pct = len(done_set) / len(raw.agents)
            print(f"  [{env_idx+1}/{total}]  fp={fp}  agents={len(raw.agents)}  "
                  f"DLA_done={len(done_set)}/{len(raw.agents)} ({done_pct:.0%})  "
                  f"({elapsed:.1f}s)")
    
    # Save cache
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    
    elapsed = time.perf_counter() - t0
    if verbose:
        avg_done = sum(len(d['done_set']) for d in cache.values()) / max(len(cache), 1)
        print(f"\n[DLA-Baseline] DONE in {elapsed:.1f}s")
        print(f"  Cache size: {len(cache)} unique envs")
        print(f"  Avg DLA done: {avg_done:.2f} agents per env")
        print(f"  Saved: {cache_path}")
    
    return cache

