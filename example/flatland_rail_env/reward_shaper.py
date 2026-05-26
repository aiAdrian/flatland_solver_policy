# =============================================================================
# reward_shaper.py
# -----------------------------------------------------------------------------
# Sparse, additive reward shaper for Flatland MARL training.
# References:
#   - Ng, Harada, Russell (1999): "Policy Invariance Under Reward Transformations"
#   - Laurent et al. (2021), Flatland Competition (arXiv:2103.16511)
# Design choices:
#   - Additive components (not overwriting) — see Bug-Fix from previous stack.
#   - No idle-penalty: STOP must remain a viable yield-action for cooperation.
#   - Per-agent done bonus + team all-done bonus to encourage cooperation.
# =============================================================================

import numpy as np
from flatland.envs.step_utils.states import TrainState
from environment.environment import Environment
from marl_attention_temporal_observation.decision_point_utils import DecisionPointUtils


class SimpleRewardShaper:
    """Minimal additive reward shaper.

    Per agent per step:
        + done_bonus           when agent reaches target (once)
        + all_done_bonus       when ALL agents are done (once per agent)
        - step_penalty         while active (encourages goal-direction)
        - deadlock_penalty     while in local deadlock
        - final_not_solved     last 5 steps if not done
    """

    def __init__(
        self,
        step_penalty: float = 0.0,
        done_bonus: float = 5.0,
        all_done_bonus: float = 10.0,
        deadlock_penalty: float = 1.0,
        final_not_solved_penalty: float = 1.0,
    ):
        self.step_penalty = float(step_penalty)
        self.done_bonus = float(done_bonus)
        self.all_done_bonus = float(all_done_bonus)
        self.deadlock_penalty = float(deadlock_penalty)
        self.final_not_solved_penalty = float(final_not_solved_penalty)
        self._rewarded_done = {}
        self._all_done_given = False
        self._episode_deadlocks = set()

    def _reset_episode(self, env: Environment):
        n = len(env.raw_env.agents)
        self._rewarded_done = {i: False for i in range(n)}
        self._all_done_given = False
        self._episode_deadlocks = set()

    @staticmethod
    def _build_agent_map(env: Environment) -> np.ndarray:
        raw = env.raw_env
        agent_map = np.full((raw.height, raw.width), -1, dtype=np.int32)
        for a in raw.agents:
            if a.position is not None:
                agent_map[a.position] = int(a.handle)
        return agent_map

    def __call__(self, reward, terminal, info, env: Environment, actions=None):
        raw = env.raw_env
        if raw._elapsed_steps <= 1:
            self._reset_episode(env)

        shaped = dict(reward)
        agent_map = self._build_agent_map(env)
        all_done = all(a.state == TrainState.DONE for a in raw.agents)
        near_limit = raw._elapsed_steps > (raw._max_episode_steps - 5)
        active_handles = [
            int(a.handle) for a in raw.agents
            if a.state != TrainState.DONE and a.position is not None
        ]
        deadlock_check = len(active_handles) > 1
        give_team = all_done and not self._all_done_given

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

                if agent.state == TrainState.DONE and not self._rewarded_done[handle]:
                    r += self.done_bonus
                    self._rewarded_done[handle] = True

                if give_team:
                    r += self.all_done_bonus

                if near_limit and not all_done and agent.state < TrainState.DONE:
                    r -= self.final_not_solved_penalty

            shaped[handle] = float(r)

        if give_team:
            self._all_done_given = True

        return shaped
