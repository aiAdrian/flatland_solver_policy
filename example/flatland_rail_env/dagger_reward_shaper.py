"""
DAgger-style reward shaper for Flatland MAPF-RL.

Combines BC-warmstarted MAPPO with imitation maintenance:
  - Soft penalty when policy deviates from DLA's recommendation
    (only at decision points)
  - Strong penalty for entering / staying in local deadlock
  - Standard bonuses for goal-reach, progress, all-done

Philosophy: "Stay close to the DLA expert, but improve where you can.
            Heavily penalize the failure modes DLA was designed to avoid."
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np

from flatland.envs.rail_env import RailEnvActions
from flatland.envs.step_utils.states import TrainState

from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.deadlock_avoidance_policy \
    import DeadLockAvoidancePolicy

try:
    from marl_attention_temporal_observation.decision_point_utils import DecisionPointUtils
    _DP_UTILS_AVAILABLE = True
except Exception:
    _DP_UTILS_AVAILABLE = False


class DAggerRewardShaper:
    """Reward shaper that maintains DLA-imitation while allowing exploration."""

    def __init__(
        self,
        dla_deviation_penalty: float = 0.10,
        deadlock_step_penalty: float = 0.50,
        deadlock_entry_penalty: float = 5.00,
        useless_stop_penalty: float = 0.10,
        progress_bonus: float = 0.02,
        done_bonus: float = 50.0,
        all_done_bonus: float = 100.0,
        final_not_solved_penalty: float = 1.0,
        decision_points_only: bool = True,
        verbose: bool = False,
    ):
        self.dla_deviation_penalty = float(dla_deviation_penalty)
        self.deadlock_step_penalty = float(deadlock_step_penalty)
        self.deadlock_entry_penalty = float(deadlock_entry_penalty)
        self.useless_stop_penalty = float(useless_stop_penalty)
        self.progress_bonus = float(progress_bonus)
        self.done_bonus = float(done_bonus)
        self.all_done_bonus = float(all_done_bonus)
        self.final_not_solved_penalty = float(final_not_solved_penalty)
        self.decision_points_only = bool(decision_points_only)
        self.verbose = bool(verbose)

        self._dla_ref: Optional[DeadLockAvoidancePolicy] = None
        self._env = None
        self._was_in_deadlock: Dict[int, bool] = {}
        self._prev_sp_distance: Dict[int, float] = {}
        self._was_done: Dict[int, bool] = {}
        self._all_done_credited: bool = False
        self._step_count: int = 0
        self._max_episode_steps: int = 500
        self._tb_writer = None

        # Diagnostics
        self._diag_dla_calls = 0
        self._diag_dla_match = 0
        self._diag_dla_deviate_at_dp = 0
        self._diag_deadlock_entries = 0
        self._diag_total_progress = 0.0
        self._diag_total_deviation_penalty = 0.0

    def set_tensorboard_writer(self, writer):
        self._tb_writer = writer

    def reset(self, env):
        self._env = env
        raw_env = env.raw_env if hasattr(env, "raw_env") else env

        try:
            self._dla_ref = DeadLockAvoidancePolicy(
                raw_env, action_size=5, enable_eps=False
            )
            self._dla_ref.reset(env)
        except Exception as e:
            if self.verbose:
                print(f"[DAggerShaper] Failed to init DLA reference: {e}")
            self._dla_ref = None

        self._was_in_deadlock = {}
        self._prev_sp_distance = {}
        self._was_done = {}
        self._all_done_credited = False
        self._step_count = 0

        try:
            self._max_episode_steps = int(getattr(raw_env, "_max_episode_steps", 500))
        except Exception:
            self._max_episode_steps = 500

        self._diag_dla_calls = 0
        self._diag_dla_match = 0
        self._diag_dla_deviate_at_dp = 0
        self._diag_deadlock_entries = 0
        self._diag_total_progress = 0.0
        self._diag_total_deviation_penalty = 0.0

    def shape(
        self,
        env,
        rewards: Dict[int, float],
        dones: Dict[int, bool],
        info: Dict[str, Any],
        actions: Dict[int, int] = None,
        states_before: Dict[int, Any] = None,
        **kwargs,
    ) -> Dict[int, float]:
        """Apply DAgger-style shaping. Called once per env.step()."""
        if actions is None:
            actions = {}
        if states_before is None:
            states_before = {}

        raw_env = env.raw_env if hasattr(env, "raw_env") else env
        self._step_count += 1
        shaped = dict(rewards)

        agent_map = self._build_agent_map(raw_env)

        n_done_total = 0
        n_total = len(raw_env.agents)

        for handle, agent in enumerate(raw_env.agents):
            if handle not in shaped:
                shaped[handle] = 0.0

            state_name = self._agent_state_name(agent)
            if state_name == "DONE":
                n_done_total += 1
                if not self._was_done.get(handle, False):
                    shaped[handle] += self.done_bonus
                    self._was_done[handle] = True
                continue

            # 2a. DLA deviation penalty
            if self._dla_ref is not None and handle in actions:
                penalty = self._compute_dla_deviation(
                    handle, agent, raw_env, actions[handle], states_before.get(handle)
                )
                shaped[handle] += penalty
                self._diag_total_deviation_penalty += penalty

            # 2b. Deadlock penalties
            in_dl_now = self._is_local_deadlock(raw_env, agent, agent_map)
            was_dl = self._was_in_deadlock.get(handle, False)

            if in_dl_now and not was_dl:
                shaped[handle] -= self.deadlock_entry_penalty
                self._diag_deadlock_entries += 1
            if in_dl_now:
                shaped[handle] -= self.deadlock_step_penalty

            self._was_in_deadlock[handle] = in_dl_now

            # 2c. Useless STOP penalty
            if handle in actions and self._is_useless_stop(
                raw_env, agent, actions[handle]
            ):
                shaped[handle] -= self.useless_stop_penalty

            # 2d. SP-progress bonus
            progress = self._sp_progress(handle, agent, raw_env)
            if progress > 0:
                shaped[handle] += self.progress_bonus * progress
                self._diag_total_progress += progress

            # 2e. Final-step penalty if not done
            if (
                self._step_count >= self._max_episode_steps - 5
                and state_name != "DONE"
            ):
                shaped[handle] -= self.final_not_solved_penalty

        # 3. All-done bonus (one-shot, team)
        if n_done_total == n_total and not self._all_done_credited:
            for h in shaped:
                shaped[h] += self.all_done_bonus
            self._all_done_credited = True

        return shaped

    def end_episode(self, env, episode: int = 0):
        match_rate = self._diag_dla_match / max(1, self._diag_dla_calls)
        if self.verbose:
            print(
                f"[DAggerShaper] ep={episode} "
                f"dla_match={match_rate:.1%} "
                f"dp_deviations={self._diag_dla_deviate_at_dp} "
                f"deadlock_entries={self._diag_deadlock_entries} "
                f"sp_progress={self._diag_total_progress:.0f} "
                f"deviation_cost={self._diag_total_deviation_penalty:+.1f}"
            )
        if self._tb_writer is not None:
            try:
                self._tb_writer.add_scalar(
                    "shaper/dla_match_rate", match_rate, episode
                )
                self._tb_writer.add_scalar(
                    "shaper/dp_deviations", self._diag_dla_deviate_at_dp, episode
                )
                self._tb_writer.add_scalar(
                    "shaper/deadlock_entries", self._diag_deadlock_entries, episode
                )
                self._tb_writer.add_scalar(
                    "shaper/sp_progress", self._diag_total_progress, episode
                )
                self._tb_writer.add_scalar(
                    "shaper/deviation_cost",
                    self._diag_total_deviation_penalty,
                    episode,
                )
            except Exception:
                pass

    # ─── Helpers ────────────────────────────────────────────────────
    def _compute_dla_deviation(
        self, handle: int, agent, raw_env, action: int, state_before
    ) -> float:
        """Returns negative penalty if action deviates from DLA at decision point."""
        if self._dla_ref is None:
            return 0.0
        if self._agent_state_name(agent) in ("DONE", "WAITING"):
            return 0.0
        if agent.position is None:
            return 0.0

        # Filter to decision points only (recommended)
        if self.decision_points_only and _DP_UTILS_AVAILABLE:
            try:
                cell_type = DecisionPointUtils.classify_cell_type(agent, raw_env)
                if cell_type not in ("SWITCH", "MERGING", "PRE_M"):
                    return 0.0
            except Exception:
                pass

        # Query DLA for its recommendation
        try:
            dla_action = int(self._dla_ref.act(handle, state_before, eps=0.0))
        except Exception:
            return 0.0

        self._diag_dla_calls += 1
        if int(action) == dla_action:
            self._diag_dla_match += 1
            return 0.0

        # Deviation penalty
        self._diag_dla_deviate_at_dp += 1
        return -self.dla_deviation_penalty

    def _build_agent_map(self, raw_env) -> np.ndarray:
        agent_map = np.full((raw_env.height, raw_env.width), -1, dtype=np.int32)
        for a in raw_env.agents:
            if a.position is not None:
                agent_map[a.position] = int(a.handle)
        return agent_map

    def _is_local_deadlock(self, raw_env, agent, agent_map) -> bool:
        if not _DP_UTILS_AVAILABLE:
            return False
        if agent.position is None:
            return False
        if self._agent_state_name(agent) == "DONE":
            return False
        try:
            return bool(
                DecisionPointUtils.is_local_deadlock(raw_env, agent, agent_map)
            )
        except Exception:
            return False

    def _is_useless_stop(self, raw_env, agent, action: int) -> bool:
        if int(action) != int(RailEnvActions.STOP_MOVING):
            return False
        if agent.position is None or agent.direction is None:
            return False
        try:
            transitions = raw_env.rail.get_transitions(
                *agent.position, agent.direction
            )
            n_options = sum(int(x) for x in transitions)
            if n_options == 0:
                return False  # forced STOP, not useless
            # Simplified: STO
