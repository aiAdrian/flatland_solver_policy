# =============================================================================
# train_marl.py  (PATCHED)
# -----------------------------------------------------------------------------
# Applied patches:
#   T2 — DLA demo collection skips FORWARD_ONLY cells (decision-points only).
#   T4 — MAX_EPISODE_STEPS moved to top with other constants.
#   PB — Progress bar with live stats (done/deadlock/KL/eps) via tqdm fallback.
# =============================================================================

import argparse
import os
import sys
import time
import random
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for _p in (_PROJECT_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flatland.envs.rail_env import RailEnvActions
from flatland.envs.step_utils.states import TrainState

from environment.environment import Environment
from solver.flatland.flatland_solver import FlatlandSolver
from policy.policy import Policy
from policy.heuristic_policy.shortest_path_deadlock_avoidance_policy.deadlock_avoidance_policy \
    import DeadLockAvoidancePolicy

from marl_attention_temporal_observation.decision_point_observation import DecisionPointObservation
from marl_attention_temporal_observation.spawn_aware_observation import SpawnAwareObservation
from marl_attention_temporal_observation.conflict_aware_observation import ConflictAwareObservation
from marl_attention_temporal_observation.decision_point_utils import DecisionPointUtils

from rendering.flatland.flatland_simple_renderer import FlatlandSimpleRenderer  # noqa: E402


# =============================================================================
# OBSERVATION REGISTRY
# -----------------------------------------------------------------------------
# Central registry of all available observation builders.
# Each entry maps a CLI key to (builder_class, base_obs_size, description).
#
# To add a new observation:
#   1. Implement it as a subclass of DecisionPointObservation
#   2. Add a new entry below
#   3. Use --obs <key> on the CLI
# =============================================================================
OBSERVATION_REGISTRY = {
    "decision_point": {
        "class": DecisionPointObservation,
        "base_dim": DecisionPointObservation.BASE_OBS_SIZE,   # 22
        "description": "Baseline: decision-point graph + tree payload",
    },
    "spawn_aware": {
        "class": SpawnAwareObservation,
        "base_dim": SpawnAwareObservation.BASE_OBS_SIZE,      # 25
        "description": "+ 3 spawn signals (active/pending density, ready)",
    },
    "conflict_aware": {
        "class": ConflictAwareObservation,
        "base_dim": ConflictAwareObservation.BASE_OBS_SIZE,   # 39
        "description": "+ 5 global + 9 local CBS-style conflict features",
    },
}

# Default observation if --obs is not specified.
DEFAULT_OBSERVATION = "conflict_aware"

# Module-level selected observation (set by CLI in main()).
# Defaults to DEFAULT_OBSERVATION; overridable via --obs flag.
_SELECTED_OBSERVATION_KEY: str = DEFAULT_OBSERVATION


def _get_selected_observation_key() -> str:
    """Return the currently selected observation key."""
    return _SELECTED_OBSERVATION_KEY


def _set_selected_observation(key: str) -> None:
    """Set the active observation. Validates against registry."""
    global _SELECTED_OBSERVATION_KEY
    if key not in OBSERVATION_REGISTRY:
        valid = ", ".join(sorted(OBSERVATION_REGISTRY.keys()))
        raise ValueError(
            f"Unknown observation key '{key}'. Valid keys: {valid}"
        )
    _SELECTED_OBSERVATION_KEY = key


def _get_observation_class():
    """Return the class object for the selected observation."""
    return OBSERVATION_REGISTRY[_get_selected_observation_key()]["class"]


def _get_observation_base_dim() -> int:
    """Return BASE_OBS_SIZE for the selected observation."""
    return OBSERVATION_REGISTRY[_get_selected_observation_key()]["base_dim"]


def _print_observation_info() -> None:
    """Pretty-print which observation is currently active."""
    key = _get_selected_observation_key()
    entry = OBSERVATION_REGISTRY[key]
    print(f"[Obs] Selected:    {key}")
    print(f"[Obs] Class:       {entry['class'].__name__}")
    print(f"[Obs] Base dim:    {entry['base_dim']}")
    print(f"[Obs] Description: {entry['description']}")



from flatland_rail_env_persister import RailEnvironmentPersistable
from reward_shaper import (
    SimpleRewardShaper,
    DLAImitationReward,
    OutcomeBasedReward,
    CounterfactualCTDEReward,
)
from mappo_policy import MAPPOPolicy
from tb_logger import TBLogger


# =============================================================================
# CONFIG (T4 — MAX_EPISODE_STEPS at top with other constants).
# =============================================================================
N_AGENTS = 5
GRID_W = 30
GRID_H = 40
N_CITIES = 3
NUM_ENVS = 5  # envs per agent-count → total = NUM_ENVS * len(AGENT_CURRICULUM)
AGENT_CURRICULUM = [1, 2, 3, 4, 5, 7, 10, 15, 20]  # diverse training distribution
MAX_RAILS_BETWEEN_CITIES = 2
MAX_RAIL_PAIRS_IN_CITY = 2

MAX_EPISODE_STEPS = 500

EVAL_EPISODES = 20

SEED = 42
DEVICE = "cpu"

ENV_CACHE_DIR = os.path.join(_THIS_DIR, "generated_envs", "curriculum_mixed")
DLA_BASELINE_CACHE_PATH = os.path.join(_THIS_DIR, "dla_baseline_cache.pkl")
def _bc_checkpoint_path() -> str:
    """BC checkpoint path is observation-specific to avoid dim mismatches."""
    return os.path.join(_THIS_DIR, f"bc_checkpoint_{_get_selected_observation_key()}.pt")


def _mappo_checkpoint_path() -> str:
    """MAPPO checkpoint path is observation-specific to avoid dim mismatches."""
    return os.path.join(_THIS_DIR, f"mappo_checkpoint_{_get_selected_observation_key()}.pt")

# ── Existing reward components ─────────────────────────────────────────
REWARD_DONE_BONUS = 100.0
REWARD_ALL_DONE_BONUS = 200.0
REWARD_DEADLOCK_PENALTY = 0.05    # mild
REWARD_STEP_PENALTY = 0.05
REWARD_FINAL_NOT_SOLVED = 1.0     # last 5 steps if not done

# ── NEW: targeted shaping to break STOP-mode-collapse ──────────────────
# Insight: previously, step_penalty applied uniformly → agents preferred
# STOP because moving carried risk (deadlock_penalty) but standing was
# only mildly penalised. We now (a) only penalise STOP when forward is
# legal AND free ("useless STOP"), and (b) reward shortest-path progress
# to densify the reward signal that was previously sparse (goal-only).
REWARD_USELESS_STOP_PENALTY = 0.10   # cost of choosing STOP when F is free
REWARD_PROGRESS_BONUS = 0.05

# ── DAgger-style imitation maintenance ─────────────────────────────────
# Soft penalty per decision-point step where MAPPO deviates from DLA's
# recommendation. Combined with HIGHER deadlock penalty, this trains the
# policy: "Stay close to DLA, only deviate to avoid deadlock or improve."
REWARD_DLA_DEVIATION_PENALTY = 0.05

MAPPO_HIDDEN = 64
MAPPO_LR = 5e-5  # PATCHED: was 3e-4, post-BC finetune
MAPPO_GAMMA = 0.99
MAPPO_GAE_LAMBDA = 0.95
MAPPO_CLIP_EPS = 0.05   
MAPPO_ENTROPY = 0.005  # PATCHED: was 0.02, post-BC finetune
MAPPO_VALUE_COEF = 0.5
MAPPO_GRAD_CLIP = 5.0
MAPPO_PPO_EPOCHS = 4
MAPPO_BATCH_SIZE = 256
 
EPS_START = 0.05  # Lower exploration when warmstarted from BC (was 0.30)
EPS_END = 0.01
EPS_DECAY_EPISODES = 500


# =============================================================================
# Progress bar (PB) — uses tqdm if available, else fallback.
# =============================================================================
try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


class _FallbackBar:
    def __init__(self, total: int, desc: str = ""):
        self.total = total
        self.n = 0
        self.desc = desc
        self.postfix_str = ""
        self.t0 = time.perf_counter()
        self._last_print = 0.0
        print(f"[{desc}] starting ({total} iterations)")

    def update(self, k: int = 1):
        self.n += k
        now = time.perf_counter()
        if now - self._last_print > 2.0 or self.n >= self.total:
            elapsed = now - self.t0
            pct = 100.0 * self.n / max(1, self.total)
            print(f"[{self.desc}] {self.n:5d}/{self.total} ({pct:5.1f}%)  "
                  f"{self.postfix_str}  ({elapsed:.0f}s)", flush=True)
            self._last_print = now

    def set_postfix_str(self, s: str):
        self.postfix_str = s

    def close(self):
        pass


def _make_progress_bar(total: int, desc: str):
    if _TQDM_AVAILABLE:
        return _tqdm(total=total, desc=desc, dynamic_ncols=True, leave=True)
    return _FallbackBar(total=total, desc=desc)


# =============================================================================
# RandomPolicy.
# =============================================================================
class RandomPolicy(Policy):

    def __init__(self):
        super().__init__()
        self.env: Optional[Environment] = None

    def get_name(self):
        return "RandomPolicy"

    def reset(self, env: Environment):
        self.env = env

    def start_step(self, train: bool):
        pass

    def end_episode(self, train: bool):
        pass

    def act(self, handle, state, eps=0.0):
        if isinstance(state, list) and len(state) > 0:
            state = state[-1]
        if isinstance(state, (tuple, list)) and len(state) >= 1:
            base = np.asarray(state[0], dtype=np.float32).flatten()
        else:
            base = np.asarray(state, dtype=np.float32).flatten()

        agent = self.env.raw_env.agents[handle]
        if agent.state == TrainState.DONE or agent.state == TrainState.WAITING:
            return int(RailEnvActions.DO_NOTHING)
        if agent.state.is_off_map_state():
            return int(np.random.choice([
                RailEnvActions.DO_NOTHING,
                RailEnvActions.MOVE_FORWARD,
                RailEnvActions.STOP_MOVING,
            ]))

        legal = [int(RailEnvActions.STOP_MOVING)]
        if base.shape[0] >= 3:
            l_ok = float(base[0]) > 0.5
            f_ok = float(base[1]) > 0.5
            r_ok = float(base[2]) > 0.5
            n_trans = int(l_ok) + int(f_ok) + int(r_ok)
            if n_trans <= 1:
                legal.append(int(RailEnvActions.MOVE_FORWARD))
            else:
                if l_ok:
                    legal.append(int(RailEnvActions.MOVE_LEFT))
                if f_ok:
                    legal.append(int(RailEnvActions.MOVE_FORWARD))
                if r_ok:
                    legal.append(int(RailEnvActions.MOVE_RIGHT))
        else:
            legal.append(int(RailEnvActions.MOVE_FORWARD))
        return int(np.random.choice(legal))

    def step(self, handle, state, action, reward, next_state, done, agent_finished=None):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def clone(self):
        return RandomPolicy()


# =============================================================================
# DLAWrapper.
# =============================================================================
class DLAWrapper(Policy):

    def __init__(self):
        super().__init__()
        self.dla: Optional[DeadLockAvoidancePolicy] = None
        self.env: Optional[Environment] = None

    def get_name(self):
        return "DeadLockAvoidancePolicy"

    def reset(self, env: Environment):
        self.env = env
        if self.dla is None:
            self.dla = DeadLockAvoidancePolicy(env.raw_env, action_size=5, enable_eps=False)
        self.dla.reset(env)

    def start_step(self, train: bool):
        self.dla.start_step(train)

    def end_episode(self, train: bool):
        self.dla.end_episode(train)

    def act(self, handle, state, eps=0.0):
        return int(self.dla.act(handle, state, eps))

    def step(self, handle, state, action, reward, next_state, done, agent_finished=None):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def clone(self):
        return DLAWrapper()


# =============================================================================
# DLABaselineRecorder — DLA wrapper that records per-env baseline data.
# -----------------------------------------------------------------------------
# Used by `--mode eval --policy dla` to populate the DLA baseline cache
# consumed by CounterfactualCTDEReward in MARL training.
#
# Per env: records (done_set, steps_per_agent, n_agents) keyed by env fingerprint.
# =============================================================================
class DLABaselineRecorder(Policy):
    """DLA wrapper that records per-agent arrival times for baseline cache."""

    def __init__(self):
        super().__init__()
        self.dla = DLAWrapper()
        self.env: Optional[Environment] = None
        # Per-env cache: fingerprint → {done_set, steps_per_agent, n_agents}
        self.cache: Dict[int, Dict[str, Any]] = {}
        # Per-episode tracking
        self._current_step: int = 0
        self._arrival_step: Dict[int, int] = {}
        self._was_done: Dict[int, bool] = {}

    def get_name(self):
        # Same as DLA so logs/eval_summary show "DeadLockAvoidancePolicy"
        return "DeadLockAvoidancePolicy"

    def reset(self, env: Environment):
        self.env = env
        self.dla.reset(env)
        n = len(env.raw_env.agents)
        self._current_step = 0
        self._arrival_step = {}
        self._was_done = {h: False for h in range(n)}

    def start_step(self, train: bool):
        self._current_step += 1
        self.dla.start_step(train)

    def end_episode(self, train: bool):
        # Snapshot final state into cache
        from reward_shaper import CounterfactualCTDEReward
        raw = self.env.raw_env
        fp = CounterfactualCTDEReward.env_fingerprint(raw)
        max_steps = int(raw._max_episode_steps)

        # Final arrival check
        for h, agent in enumerate(raw.agents):
            if agent.state == TrainState.DONE and not self._was_done.get(h, False):
                self._arrival_step[h] = self._current_step
                self._was_done[h] = True

        done_set = {h for h in range(len(raw.agents)) if self._was_done.get(h, False)}
        steps_per_agent = {
            h: self._arrival_step.get(h, max_steps) for h in range(len(raw.agents))
        }

        self.cache[fp] = {
            'done_set': set(done_set),
            'steps_per_agent': dict(steps_per_agent),
            'n_agents': len(raw.agents),
        }

        self.dla.end_episode(train)

    def act(self, handle, state, eps=0.0):
        # Track arrivals BEFORE acting (state of THIS step)
        for h, agent in enumerate(self.env.raw_env.agents):
            if agent.state == TrainState.DONE and not self._was_done.get(h, False):
                self._arrival_step[h] = self._current_step
                self._was_done[h] = True
        return int(self.dla.act(handle, state, eps))

    def step(self, handle, state, action, reward, next_state, done, agent_finished=None):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def clone(self):
        return DLABaselineRecorder()


# =============================================================================
# Environment factory.
# =============================================================================
def build_environment(
    n_agents: int = N_AGENTS,
    grid_w: int = GRID_W,
    grid_h: int = GRID_H,
    n_cities: int = N_CITIES,
    num_envs: int = NUM_ENVS,
    env_cache_dir: str = ENV_CACHE_DIR,
    seed: int = SEED,
) -> Environment:

    def obs_builder_factory():
        """Instantiate the observation builder selected via --obs CLI flag."""
        obs_class = _get_observation_class()
        return obs_class(verbose_first_call=False)

    env = RailEnvironmentPersistable(
        obs_builder_object_creator=obs_builder_factory,
        number_of_agents=n_agents,
        grid_width=grid_w,
        grid_height=grid_h,
        n_cities=n_cities,
        max_rails_between_cities=MAX_RAILS_BETWEEN_CITIES,
        max_rails_in_city=MAX_RAIL_PAIRS_IN_CITY * 2,
        random_seed=seed,
        disable_mal_functions=True,
        extra_line_off=True
    )

    os.makedirs(env_cache_dir, exist_ok=True)
    n_cached = len(os.listdir(env_cache_dir)) if os.path.isdir(env_cache_dir) else 0

    # Use module-level AGENT_CURRICULUM
    expected_total = num_envs * len(AGENT_CURRICULUM)
    if n_cached < expected_total:
        print(f"[Env] Generating {expected_total} envs ({num_envs} per agent-count) "
              f"to {env_cache_dir} ...")
        print(f"[Env] Agent counts: {AGENT_CURRICULUM}")
        env.generate_and_persist_environments(
            generate_nbr_env=num_envs,
            generate_agents_per_env=AGENT_CURRICULUM,
            path=env_cache_dir,
            overwrite_existing=False,
        )
    else:
        print(f"[Env] Using {n_cached} cached environments at {env_cache_dir}")

    env._loaded_env = []
    env._loaded_env_itr = 0
    env.load_environments_from_path(path=env_cache_dir)

    _original_reset = env.reset

    def _patched_reset(*args, **kwargs):
        result = _original_reset(*args, **kwargs)
        env.raw_env._max_episode_steps = MAX_EPISODE_STEPS
        return result

    env.reset = _patched_reset
    return env

# =============================================================================
# Solver setup.
# =============================================================================
def setup_solver(env: Environment, policy: Policy, do_render: bool = False) -> FlatlandSolver:
    solver = FlatlandSolver(env, 
                            policy,
                            FlatlandSimpleRenderer(env) if do_render else None
        )
 

    if hasattr(solver, "set_max_steps"):
        solver.set_max_steps(MAX_EPISODE_STEPS)

    reward_kind = os.environ.get("REWARD_KIND", "outcome_based").lower()
    shaper = None  # initialized below based on final reward_kind

    if reward_kind == "counterfactual_ctde":
        if not os.path.exists(DLA_BASELINE_CACHE_PATH):
            print(f"[reward] WARNING: no DLA cache at {DLA_BASELINE_CACHE_PATH}")
            print(f"[reward] Run: python train_marl.py --mode precompute_dla")
            print(f"[reward] Falling back to OutcomeBasedReward...")
            reward_kind = "outcome_based"
        else:
            import pickle
            with open(DLA_BASELINE_CACHE_PATH, 'rb') as f:
                dla_cache = pickle.load(f)
            shaper = CounterfactualCTDEReward(
                all_done_bonus=2.0,
                step_penalty=0.01,
                reward_scale=1.0,
                dla_cache=dla_cache,
                verbose=False,
            )
            print(f"[reward] Using CounterfactualCTDEReward "
                  f"(CTDE, {len(dla_cache)} DLA cache entries)")

    if reward_kind == "outcome_based" and shaper is None:
        shaper = OutcomeBasedReward(
            step_penalty=1.5,         # PATCHED: was 1.0 → mehr Speed-Pressure
            deadlock_penalty=5.0,     # -5 per step in deadlock
            done_bonus=50.0,          # +50 flat on arrival
            time_saved_factor=2.0,    # PATCHED: was 1.0 → Bonus für kurze Eps verdoppeln
            all_done_bonus=100.0,     # +100 per agent if all done (team!)
            fail_penalty=200.0,       # -200 per agent not done
            match_bonus=0.5,          # tiny DLA hint at decision points
            reward_scale=0.01,        # PPO stability
        )
        print("[reward] Using OutcomeBasedReward "
              "(step=-1, deadlock=-5, done=+50+saved, all_done=+100, fail=-200, scale=0.01)")
    elif reward_kind == "dla_imitation":
        shaper = DLAImitationReward(
            match_bonus=1.0,
            deviation_penalty=-1.0,
            all_done_bonus=10.0,
            fail_extra_offset=10.0,
            reward_scale=0.05,
        )
        print("[reward] Using DLAImitationReward "
              "(match=+1, deviate=-1, all_done=+10, fail=-(max+10), scale=0.05)")
    elif shaper is None:
        shaper = SimpleRewardShaper(
            step_penalty=REWARD_STEP_PENALTY,
            done_bonus=REWARD_DONE_BONUS,
            all_done_bonus=REWARD_ALL_DONE_BONUS,
            deadlock_penalty=REWARD_DEADLOCK_PENALTY,
            final_not_solved_penalty=REWARD_FINAL_NOT_SOLVED,
            useless_stop_penalty=REWARD_USELESS_STOP_PENALTY,
            progress_bonus=REWARD_PROGRESS_BONUS,
            dla_deviation_penalty=REWARD_DLA_DEVIATION_PENALTY,
            dla_decay_steps=200,
            done_decay_steps=0,
            stop_yields_to_dla_forward=True,
        )
        print("[reward] Using SimpleRewardShaper (legacy)")

    solver.set_reward_shaper(shaper)
    return solver


def _count_deadlock_rate(env: Environment) -> float:
    raw = env.raw_env
    n = len(raw.agents)
    if n == 0:
        return 0.0
    agent_map = np.full((raw.height, raw.width), -1, dtype=np.int32)
    for a in raw.agents:
        if a.position is not None:
            agent_map[a.position] = int(a.handle)
    n_deadlock = 0
    for a in raw.agents:
        if a.position is not None and a.state != TrainState.DONE:
            if DecisionPointUtils.is_local_deadlock(raw, a, agent_map):
                n_deadlock += 1
    return n_deadlock / n


# =============================================================================
# Eval loop (PB).
# =============================================================================
def run_eval(
    policy: Policy,
    n_episodes: int = EVAL_EPISODES,
    seed_offset: int = 10000,
    verbose: bool = True,
    tb_logger: Optional["TBLogger"] = None,
    do_render: bool = False,
    save_baseline_cache: Optional[str] = None,
) -> Dict[str, float]:
    env = build_environment()
    solver = setup_solver(env, policy, do_render)

    done_rates: List[float] = []
    deadlock_rates: List[float] = []
    rewards: List[float] = []
    steps_list: List[int] = []

    print(f"\n[Eval] {policy.get_name()}  x  {n_episodes} episodes")
    print("-" * 60)
    t0 = time.perf_counter()

    bar = _make_progress_bar(total=n_episodes, desc=f"Eval[{policy.get_name()}]")

    # Schedule a feature-report at the LAST eval episode.
    # The observation builder consumes _force_feature_report at episode end.
    # We arm it at episode N-1 so it fires when the very last episode flushes.
    try:
        from marl_attention_temporal_observation.decision_point_observation import (
            DecisionPointObservation,
        )
        _force_obs_class = DecisionPointObservation
    except Exception:
        _force_obs_class = None

    for ep in range(n_episodes):
        # Arm force-flag for the LAST episode.
        if _force_obs_class is not None and ep == n_episodes - 1:
            _force_obs_class._force_feature_report = True
        ep_seed = seed_offset + ep
        np.random.seed(ep_seed)
        random.seed(ep_seed)
        torch.manual_seed(ep_seed)

        tot_reward, tot_terminate, tot_steps = solver.run_episode(
            episode=ep,
            env=env,
            policy=policy,
            eps=0.0,
            training_mode=False,
        )

        done_rates.append(float(tot_terminate))
        rewards.append(float(tot_reward))
        steps_list.append(int(tot_steps))
        deadlock_rates.append(_count_deadlock_rate(env))

        n_agents_ep = len(env.raw_env.agents)
        if tb_logger is not None:
            tb_logger.log_eval_episode(
                step=ep,
                done_rate=float(np.mean(done_rates)),
                deadlock_rate=float(np.mean(deadlock_rates)),
                episode_len=float(np.mean(steps_list)),
                total_reward=float(np.mean(rewards)),
            )
            tb_logger.log_scalar("env/n_agents", n_agents_ep, ep)
            tb_logger.log_scalar("env/done_count", int(round(float(tot_terminate) * n_agents_ep)), ep)
            # DLA-deviation stats → TensorBoard
            _shaper = getattr(solver, '_reward_shaper', None) or getattr(solver, 'reward_shaper', None)
            if _shaper is not None:
                _calls = getattr(_shaper, '_diag_dla_calls', 0)
                if _calls > 0:
                    _dev = _shaper._diag_dla_deviate
                    _rate = _dev / max(1, _calls)
                    tb_logger.log_scalar("shaper/dla_dev_rate", _rate, ep)
                    tb_logger.log_scalar("shaper/dla_deviate_count", _dev, ep)
                    tb_logger.log_scalar("shaper/dla_calls", _calls, ep)

        # Bar visualization based on running done-rate
        BAR_LEN = 25
        current_done = float(np.mean(done_rates))
        current_dlk = float(np.mean(deadlock_rates))
        current_len = float(np.mean(steps_list))
        current_rew = float(np.mean(rewards))
        b = int(np.round(BAR_LEN * current_done))
        done_bar = '#' * b + '_' * (BAR_LEN - b)

        # This-episode agent count (real, not hardcoded)
        ep_done_n = int(round(float(tot_terminate) * n_agents_ep))

        # DLA-deviation stats from the reward shaper (DAgger)
        shaper = getattr(solver, '_reward_shaper', None) or getattr(solver, 'reward_shaper', None)
        dla_str = ""
        if shaper is not None:
            calls = getattr(shaper, '_diag_dla_calls', 0)
            if calls > 0:
                deviate = shaper._diag_dla_deviate
                dev_rate = deviate / max(1, calls)
                dla_str = f"  DLA[dev:{deviate:>3d}/{calls:>3d} ({dev_rate:>3.0%})]"

        bar.set_postfix_str(
            f"agents:[{ep_done_n:>2d}/{n_agents_ep:>2d}]  "
            f"done={current_done:>4.0%} "
            f"[{done_bar}]  "
            f"deadlock={current_dlk:>4.0%}  "
            f"steps={current_len:>4.0f}  "
            f"reward={current_rew:>+8.1f}"
            f"{dla_str}"
        )

        bar.update(1)

    bar.close()
    elapsed = time.perf_counter() - t0
    mean_done = float(np.mean(done_rates)) if done_rates else 0.0
    std_done = float(np.std(done_rates)) if done_rates else 0.0
    mean_dl = float(np.mean(deadlock_rates)) if deadlock_rates else 0.0
    mean_len = float(np.mean(steps_list)) if steps_list else 0.0
    mean_rew = float(np.mean(rewards)) if rewards else 0.0

    print("-" * 60)
    print(f"[Eval]  {policy.get_name()}  RESULT  ({elapsed:.1f}s)")
    print(f"  done_rate     = {mean_done:.3f}  (+/-{std_done:.3f})")
    print(f"  deadlock_rate = {mean_dl:.3f}")
    print(f"  episode_len   = {mean_len:.1f}")
    print(f"  total_reward  = {mean_rew:+.2f}")
    print()

    if tb_logger is not None:
        tb_logger.log_eval_summary(
            done_rate=mean_done,
            deadlock_rate=mean_dl,
            episode_len=mean_len,
            total_reward=mean_rew,
            n_episodes=len(done_rates),
        )

    # Save baseline cache if requested (DLA-eval used as MARL counterfactual baseline)
    if save_baseline_cache is not None and isinstance(policy, DLABaselineRecorder):
        import pickle
        os.makedirs(os.path.dirname(save_baseline_cache) or ".", exist_ok=True)
        with open(save_baseline_cache, 'wb') as f:
            pickle.dump(policy.cache, f)
        print(f"[Eval] Saved DLA baseline cache: {save_baseline_cache} "
              f"({len(policy.cache)} unique envs)")

    return {
        "done_rate": mean_done,
        "done_std": std_done,
        "deadlock_rate": mean_dl,
        "episode_len": mean_len,
        "total_reward": mean_rew,
        "n_episodes": len(done_rates),
    }

# =============================================================================
# DLA recording wrapper (T2 — decision-points only).
# =============================================================================
class DLARecordingWrapper(Policy):

    def __init__(self):
        super().__init__()
        self.dla = DLAWrapper()
        self.env: Optional[Environment] = None
        self.demos: List[Tuple] = []
        self._calls_total = 0
        self._calls_recorded = 0

    def get_name(self):
        return "DLARecordingWrapper"

    def reset(self, env):
        self.env = env
        self.dla.reset(env)

    def start_step(self, train: bool):
        self.dla.start_step(train)

    def end_episode(self, train: bool):
        self.dla.end_episode(train)

    def act(self, handle, state, eps=0.0):
        action = int(self.dla.act(handle, state, eps))
        self._calls_total += 1

        try:
            agent = self.env.raw_env.agents[handle]
            cell_type = DecisionPointUtils.classify_cell_type(agent, self.env.raw_env)
        except Exception:
            cell_type = "OUTSIDE"

        if cell_type not in ("SWITCH", "MERGING", "PRE_M"):
            return action

        base_obs, opps, payload = MAPPOPolicy._unwrap_state(state)
        bd = _get_observation_base_dim()
        self.demos.append((
            base_obs[:bd].copy(),
            [np.asarray(o, dtype=np.float32).flatten()[:bd].copy() for o in opps],
            payload,
            action,
        ))
        self._calls_recorded += 1
        return action

    def step(self, handle, state, action, reward, next_state, done, agent_finished=None):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def clone(self):
        return DLARecordingWrapper()


def collect_dla_demos(n_episodes: int = 100, verbose: bool = True) -> List[Tuple]:
    env = build_environment()
    recorder = DLARecordingWrapper()
    solver = setup_solver(env, recorder)

    print(f"\n[DemoCollect] DLA x {n_episodes} episodes (decision-points only) ...")
    t0 = time.perf_counter()

    bar = _make_progress_bar(total=n_episodes, desc="DemoCollect")
    for ep in range(n_episodes):
        np.random.seed(20000 + ep)
        random.seed(20000 + ep)
        solver.run_episode(
            episode=ep,
            env=env,
            policy=recorder,
            eps=0.0,
            training_mode=False,
        )
        bar.set_postfix_str(f"demos={len(recorder.demos)}")
        bar.update(1)
    bar.close()

    elapsed = time.perf_counter() - t0
    keep_rate = (recorder._calls_recorded / max(1, recorder._calls_total))
    print(f"[DemoCollect] {len(recorder.demos)} demos in {elapsed:.1f}s "
          f"(decision-point keep-rate={keep_rate:.1%})")
    return recorder.demos

# =============================================================================
# BC pretraining mode.
# =============================================================================
def run_bc(n_demo_episodes: int = 200, n_bc_epochs: int = 10):
    print("\n" + "=" * 70)
    print("BEHAVIOR CLONING from DLA expert (decision-points only)")
    print("=" * 70)
    bc_logger = TBLogger(run_name=f"bc_{_get_selected_observation_key()}")

    demos = collect_dla_demos(n_episodes=n_demo_episodes)
    if len(demos) == 0:
        print("[BC] No demos collected — aborting.")
        bc_logger.close()
        return

    policy = MAPPOPolicy(
        base_dim=_get_observation_base_dim(),
        hidden=MAPPO_HIDDEN,
        learning_rate=MAPPO_LR,
        gamma=MAPPO_GAMMA,
        gae_lambda=MAPPO_GAE_LAMBDA,
        clip_eps=MAPPO_CLIP_EPS,
        entropy_coef=MAPPO_ENTROPY,
        value_coef=MAPPO_VALUE_COEF,
        max_grad_norm=MAPPO_GRAD_CLIP,
        ppo_epochs=MAPPO_PPO_EPOCHS,
        batch_size=MAPPO_BATCH_SIZE,
        device=DEVICE,
    )

    stats = policy.train_bc(demos, n_epochs=n_bc_epochs, batch_size=MAPPO_BATCH_SIZE)
    bc_ckpt = _bc_checkpoint_path()
    policy.save(bc_ckpt)

    print(f"\n[BC] Final  loss={stats['bc_loss']:.4f}  acc={stats['bc_acc']:.3f}")
    print(f"[BC] Checkpoint saved: {bc_ckpt}")

    bc_logger.log_bc_epoch(epoch=n_bc_epochs, loss=stats["bc_loss"], accuracy=stats["bc_acc"])

    print(f"\n[BC] Evaluating BC-pretrained policy on {EVAL_EPISODES} episodes ...")
    metrics = run_eval(policy, n_episodes=EVAL_EPISODES, verbose=False, tb_logger=bc_logger)
    print(f"[BC] BC-policy done_rate = {metrics['done_rate']:.3f}")
    bc_logger.close()


# =============================================================================
# MAPPO training mode (PB).
# =============================================================================
def run_train(n_episodes: int = 2000):
    print("\n" + "=" * 70)
    print(f"MAPPO TRAINING  ({n_episodes} episodes)")
    print("=" * 70)

    env = build_environment()
    policy = MAPPOPolicy(
        base_dim=_get_observation_base_dim(),
        hidden=MAPPO_HIDDEN,
        learning_rate=MAPPO_LR,
        gamma=MAPPO_GAMMA,
        gae_lambda=MAPPO_GAE_LAMBDA,
        clip_eps=MAPPO_CLIP_EPS,
        entropy_coef=MAPPO_ENTROPY,
        value_coef=MAPPO_VALUE_COEF,
        max_grad_norm=MAPPO_GRAD_CLIP,
        ppo_epochs=MAPPO_PPO_EPOCHS,
        batch_size=MAPPO_BATCH_SIZE,
        device=DEVICE,
    )

    mappo_ckpt = _mappo_checkpoint_path()
    bc_ckpt = _bc_checkpoint_path()
    if os.path.exists(mappo_ckpt):
        print(f"[Train] Resuming from MAPPO checkpoint: {mappo_ckpt}")
        policy.load(mappo_ckpt)
    elif os.path.exists(bc_ckpt):
        print(f"[Train] Loading BC warmstart from {bc_ckpt}")
        policy.load(bc_ckpt)
        policy.episode_count = 0
    else:
        print("[Train] No warmstart — training from scratch.")


    solver = setup_solver(env, policy)
    train_logger = TBLogger(run_name=f"train_mappo_{_get_selected_observation_key()}")

    t0 = time.perf_counter()
    done_history: List[float] = []
    reward_history: List[float] = []
    eval_log: List[Dict[str, Any]] = []

    bar = _make_progress_bar(total=n_episodes, desc="Train[MAPPO]")

    for ep in range(n_episodes):
        progress = min(1.0, ep / max(1, EPS_DECAY_EPISODES))
        eps = EPS_START + (EPS_END - EPS_START) * progress
         
        tot_reward, tot_terminate, tot_steps = solver.run_episode(
            episode=ep,
            env=env,
            policy=policy,
            eps=eps,
            training_mode=True,
        )
        done_history.append(float(tot_terminate))
        reward_history.append(float(tot_reward))

        recent_done_now = float(np.mean(done_history[-50:]))
        recent_rew_now = float(np.mean(reward_history[-50:]))

        n_agents_ep = len(env.raw_env.agents)
        train_logger.log_train_episode(
            episode=ep,
            done_50=recent_done_now,
            reward_50=recent_rew_now,
            eps=eps,
            ppo_stats=policy.last_train_stats,
        )
        train_logger.log_scalar("env/n_agents", n_agents_ep, ep)
        train_logger.log_scalar("env/done_count", int(round(float(tot_terminate) * n_agents_ep)), ep)

        # DLA-deviation stats → TensorBoard
        _shaper = getattr(solver, '_reward_shaper', None) or getattr(solver, 'reward_shaper', None)
        if _shaper is not None:
            _calls = getattr(_shaper, '_diag_dla_calls', 0)
            if _calls > 0:
                _dev = _shaper._diag_dla_deviate
                _rate = _dev / max(1, _calls)
                _cost = -_dev * _shaper.dla_deviation_penalty
                train_logger.log_scalar("shaper/dla_dev_rate", _rate, ep)
                train_logger.log_scalar("shaper/dla_deviate_count", _dev, ep)
                train_logger.log_scalar("shaper/dla_calls", _calls, ep)
                train_logger.log_scalar("shaper/dla_penalty_cost", _cost, ep)

        # Action distribution → TensorBoard
        _ad = getattr(policy, 'last_action_dist', None)
        if _ad is not None:
            train_logger.log_scalar("actions/do_nothing",  float(_ad.get(0, 0.0)), ep)
            train_logger.log_scalar("actions/move_left",   float(_ad.get(1, 0.0)), ep)
            train_logger.log_scalar("actions/move_forward",float(_ad.get(2, 0.0)), ep)
            train_logger.log_scalar("actions/move_right",  float(_ad.get(3, 0.0)), ep)
            train_logger.log_scalar("actions/stop",        float(_ad.get(4, 0.0)), ep)

        # Bar visualization based on running done-rate
        BAR_LEN = 25
        b = int(np.round(BAR_LEN * recent_done_now))
        done_bar = '#' * b + '_' * (BAR_LEN - b)

        stats = policy.last_train_stats

        # This-episode agent count (real, not hardcoded)
        ep_done_n = int(round(float(tot_terminate) * n_agents_ep))

        # DLA-deviation stats from the reward shaper (DAgger)
        shaper = getattr(solver, '_reward_shaper', None) or getattr(solver, 'reward_shaper', None)
        dla_str = ""
        if shaper is not None:
            calls = getattr(shaper, '_diag_dla_calls', 0)
            if calls > 0:
                deviate = shaper._diag_dla_deviate
                dev_rate = deviate / max(1, calls)
                dla_str = f"  DLA[dev:{deviate:>3d}/{calls:>3d} ({dev_rate:>3.0%})]"

        # Action distribution: [DO_NOTHING, LEFT, FORWARD, RIGHT, STOP]
        ad = getattr(policy, 'last_action_dist', {0: 0, 1: 0, 2: 0, 3: 0, 4: 0})
        act_str = (f"acts[N:{ad.get(0,0):.0%} "
                   f"L:{ad.get(1,0):.0%} "
                   f"F:{ad.get(2,0):.0%} "
                   f"R:{ad.get(3,0):.0%} "
                   f"S:{ad.get(4,0):.0%}]")

        bar.set_postfix_str(
            f"agents:[{ep_done_n:>2d}/{n_agents_ep:>2d}]  "
            f"done={recent_done_now:>4.0%} "
            f"[{done_bar}]  "
            f"reward={recent_rew_now:>+6.0f}  "
            f"eps={eps:>4.0%}  "
            f"KL={stats.get('kl', 0):>+6.3f}  "
            f"H={stats.get('ent', 0):>4.2f}  "
            f"{act_str}"
            f"{dla_str}"
        )


        bar.update(1)


        if (ep + 1) % 100 == 0:
            print(f"\n[Train] Mid-training eval at episode {ep+1} ...")
            eval_metrics = run_eval(policy, n_episodes=10, verbose=False, tb_logger=train_logger)
            eval_log.append({"episode": ep + 1, **eval_metrics})
            policy.save(_mappo_checkpoint_path())

    bar.close()
    policy.save(_mappo_checkpoint_path())
    elapsed = time.perf_counter() - t0
    print(f"\n[Train] Training complete in {elapsed/60:.1f} min")
    print("[Train] Final eval ...")
    final_metrics = run_eval(policy, n_episodes=EVAL_EPISODES, verbose=False, tb_logger=train_logger)
    eval_log.append({"episode": n_episodes, **final_metrics})
    train_logger.close()

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    for entry in eval_log:
        print(f"  ep {entry['episode']:5d}: done={entry['done_rate']:.3f}  "
              f"deadlock={entry['deadlock_rate']:.3f}  rew={entry['total_reward']:+.1f}")

# =============================================================================
# Eval-only mode.
# =============================================================================
def run_eval_mode(policy_name: str, n_episodes: int = EVAL_EPISODES, do_render: bool = False):
    print("\n" + "=" * 70)
    print(f"EVAL MODE  ({policy_name}, {n_episodes} episodes)")
    print("=" * 70)

    if policy_name == "random":
        policy = RandomPolicy()
    elif policy_name == "dla":
        # Use Recorder variant: tracks per-env (done_set, steps_per_agent)
        # and auto-saves to DLA_BASELINE_CACHE_PATH for use by CFReward.
        policy = DLABaselineRecorder()
    elif policy_name == "mappo":
        policy = MAPPOPolicy(
            base_dim=_get_observation_base_dim(),
            hidden=MAPPO_HIDDEN,
            device=DEVICE,
        )
        mappo_ckpt = _mappo_checkpoint_path()
        bc_ckpt = _bc_checkpoint_path()
        ckpt = mappo_ckpt if os.path.exists(mappo_ckpt) else bc_ckpt
        if os.path.exists(ckpt):
            policy.load(ckpt)
        else:
            print(f"[Eval] WARNING: No checkpoint at {mappo_ckpt} or {bc_ckpt}")
            print("[Eval] Evaluating random-init MAPPO (expect very low done-rate).")
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    logger = TBLogger(run_name=f"eval_{policy_name}_{_get_selected_observation_key()}")
    # Save cache ONLY when explicitly enabled (e.g. by precompute_dla mode).
    # Prevents accidental overwrites during regular eval runs.
    save_cache = None
    if policy_name == "dla" and os.environ.get("SAVE_DLA_CACHE") == "1":
        save_cache = DLA_BASELINE_CACHE_PATH
        print(f"[Eval] DLA + SAVE_DLA_CACHE=1 → will save cache to {save_cache}")
    elif policy_name == "dla":
        print(f"[Eval] DLA mode (read-only — set SAVE_DLA_CACHE=1 to write cache)")
    try:
        return run_eval(
            policy,
            n_episodes=n_episodes,
            tb_logger=logger,
            do_render=do_render,
            save_baseline_cache=save_cache,
        )
    finally:
        logger.close()


# =============================================================================
# CLI.
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Flatland MARL — simple training/eval",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--mode", choices=["eval", "bc", "train", "precompute_dla"], required=True)
    parser.add_argument("--policy", choices=["random", "dla", "mappo"], default="mappo")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--bc-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        '--render',
        action='store_true',
        dest='render',
        help='Enable FlatlandSimpleRenderer during evaluation/training runs'
    )
    # Observation selector — choose which observation builder to use.
    obs_help_lines = ["Observation builder to use:"]
    for key, entry in sorted(OBSERVATION_REGISTRY.items()):
        marker = " (default)" if key == DEFAULT_OBSERVATION else ""
        obs_help_lines.append(
            f"  {key:<16} dim={entry['base_dim']:<3} {entry['description']}{marker}"
        )
    parser.add_argument(
        "--obs",
        choices=sorted(OBSERVATION_REGISTRY.keys()),
        default=DEFAULT_OBSERVATION,
        help="\n".join(obs_help_lines),
    )

    args = parser.parse_args()

    # Activate selected observation BEFORE any environment is built.
    _set_selected_observation(args.obs)
    _print_observation_info()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("\n" + "=" * 60)
    print(f"  Flatland MARL — mode={args.mode}")
    print("=" * 60)

    if args.mode == "eval":
        n_ep = args.episodes if args.episodes is not None else EVAL_EPISODES
        do_render = args.render if args.render is not None else False
        run_eval_mode(args.policy, n_episodes=n_ep, do_render=do_render)
    elif args.mode == "bc":
        n_demo_ep = args.episodes if args.episodes is not None else 200
        run_bc(n_demo_episodes=n_demo_ep, n_bc_epochs=args.bc_epochs)
    elif args.mode == "precompute_dla":
        # Run DLA over EVERY cached env. Build env FIRST (triggers generation
        # if needed), THEN count what's actually on disk. --episodes is ignored.
        from glob import glob

        # Trigger generation if cache is empty
        _tmp_env = build_environment()
        del _tmp_env

        n_cached = len(glob(os.path.join(ENV_CACHE_DIR, "*.pkl")))
        if n_cached == 0:
            print(f"[precompute_dla] ERROR: no envs in {ENV_CACHE_DIR}")
            sys.exit(1)

        if args.episodes is not None:
            print(f"[precompute_dla] Note: --episodes {args.episodes} IGNORED. "
                  f"Running DLA over all {n_cached} cached envs.")
        print(f"[precompute_dla] Running DLA over {n_cached} cached envs ...")
        os.environ["SAVE_DLA_CACHE"] = "1"
        try:
            run_eval_mode("dla", n_episodes=n_cached)
        finally:
            os.environ.pop("SAVE_DLA_CACHE", None)
    elif args.mode == "train":
        n_ep = args.episodes if args.episodes is not None else 2000
        run_train(n_episodes=n_ep)


if __name__ == "__main__":
    main()



'''
>> Baseline : DeadLockAvoidancePolicy

=========================================================================================================
Cell-type distribution over full DLA episode
=========================================================================================================
Total agent-step samples: 255
  FORWARD_ONLY   :   127 ( 49.8%)
  OUTSIDE        :    44 ( 17.3%)
  SWITCH         :    33 ( 12.9%)
  DONE           :    31 ( 12.2%)
  MERGING        :    20 (  7.8%)
=========================================================================================================
Cell-type distribution at DLA decision points
=========================================================================================================


>> Baseline : DeadLockAvoidancePolicy

=========================================================================================================
DLA ACTION DISTRIBUTION  (50 ep, 19111 samples)
=========================================================================================================
CELL_TYPE           DO_NOTHING      MOVE_LEFT      MOVE_FORWARD     MOVE_RIGHT      STOP_MOVING     TOTAL
---------------------------------------------------------------------------------------------------------
SWITCH               0 (  0.0%)    300 ( 14.5%)   1480 ( 71.4%)    256 ( 12.3%)     37 (  1.8%)      2073
MERGING              0 (  0.0%)      0 (  0.0%)    938 ( 40.0%)      0 (  0.0%)   1409 ( 60.0%)      2347
FORWARD_ONLY         0 (  0.0%)      0 (  0.0%)  11535 ( 81.3%)      0 (  0.0%)   2660 ( 18.7%)     14195
OUTSIDE              0 (  0.0%)      0 (  0.0%)    257 ( 51.8%)      0 (  0.0%)    239 ( 48.2%)       496

=========================================================================================================
DECISION POINTS ONLY (SWITCH + MERGING + PRE_M)
=========================================================================================================

'''