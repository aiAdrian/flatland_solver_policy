# =============================================================================
# train_marl.py
# -----------------------------------------------------------------------------
# Main training/evaluation script for Flatland MARL.
#
# Modes:
#   --mode eval --policy {random|dla|mappo}   [--episodes N]
#   --mode bc   [--episodes 200] [--bc-epochs 10]
#   --mode train [--episodes 2000]
# =============================================================================

import argparse
import os
import sys
import time
import random
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch

# --- Path setup --------------------------------------------------------------
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
from marl_attention_temporal_observation.decision_point_utils import DecisionPointUtils

from flatland_rail_env_persister import RailEnvironmentPersistable
from reward_shaper import SimpleRewardShaper
from mappo_policy import MAPPOPolicy
from tb_logger import TBLogger


# =============================================================================
# CONFIG.
# =============================================================================
N_AGENTS = 5
GRID_W = 30
GRID_H = 40
N_CITIES = 3
NUM_ENVS = 18
MAX_RAILS_BETWEEN_CITIES = 2
MAX_RAIL_PAIRS_IN_CITY = 2

EVAL_EPISODES = 20
SEED = 42
DEVICE = "cpu"

ENV_CACHE_DIR = os.path.join(_THIS_DIR, "generated_envs", "simple_marl")
BC_CHECKPOINT = os.path.join(_THIS_DIR, "bc_checkpoint.pt")
MAPPO_CHECKPOINT = os.path.join(_THIS_DIR, "mappo_checkpoint.pt")

REWARD_DONE_BONUS = 5.0
REWARD_ALL_DONE_BONUS = 10.0
REWARD_DEADLOCK_PENALTY = 1.0
REWARD_STEP_PENALTY = 0.0
REWARD_FINAL_NOT_SOLVED = 1.0

MAPPO_HIDDEN = 64
MAPPO_LR = 3e-4
MAPPO_GAMMA = 0.99
MAPPO_GAE_LAMBDA = 0.95
MAPPO_CLIP_EPS = 0.20
MAPPO_ENTROPY = 0.02
MAPPO_VALUE_COEF = 0.5
MAPPO_GRAD_CLIP = 0.5
MAPPO_PPO_EPOCHS = 2
MAPPO_BATCH_SIZE = 256

EPS_START = 0.30
EPS_END = 0.05
EPS_DECAY_EPISODES = 800


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
                if l_ok: legal.append(int(RailEnvActions.MOVE_LEFT))
                if f_ok: legal.append(int(RailEnvActions.MOVE_FORWARD))
                if r_ok: legal.append(int(RailEnvActions.MOVE_RIGHT))
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
        return DecisionPointObservation()

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
    )

    os.makedirs(env_cache_dir, exist_ok=True)
    n_cached = len(os.listdir(env_cache_dir)) if os.path.isdir(env_cache_dir) else 0

    if n_cached < num_envs:
        print(f"[Env] Generating {num_envs} environments to {env_cache_dir} ...")
        env.generate_and_persist_environments(
            generate_nbr_env=num_envs,
            generate_agents_per_env=[n_agents] * num_envs,
            path=env_cache_dir,
            overwrite_existing=False,
        )
    else:
        print(f"[Env] Using {n_cached} cached environments at {env_cache_dir}")

    env._loaded_env = []
    env._loaded_env_itr = 0
    env.load_environments_from_path(path=env_cache_dir)

    # ------------------------------------------------------------------
    # Patch reset() so _max_episode_steps is forced after each load.
    # The persisted envs have a stale _max_episode_steps=70 baked in;
    # _copy_attribute_from_env() overwrites it on every reset, so we
    # reapply ours AFTER reset to guarantee episodes are long enough.
    # ------------------------------------------------------------------
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
# Standard Flatland formula: 4 * (2*(W+H) + n_cities*8)
# For 30x40 grid with 3 cities → 4 * (140 + 24) = 656 steps
MAX_EPISODE_STEPS = 500  # capped for faster training (was 656); 400 was too short


def setup_solver(env: Environment, policy: Policy) -> FlatlandSolver:
    solver = FlatlandSolver(env, policy)

    # Try the official knob first (still useful for solver-internal step counters).
    if hasattr(solver, "set_max_steps"):
        solver.set_max_steps(MAX_EPISODE_STEPS)

    shaper = SimpleRewardShaper(
        step_penalty=REWARD_STEP_PENALTY,
        done_bonus=REWARD_DONE_BONUS,
        all_done_bonus=REWARD_ALL_DONE_BONUS,
        deadlock_penalty=REWARD_DEADLOCK_PENALTY,
        final_not_solved_penalty=REWARD_FINAL_NOT_SOLVED,
    )
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
# Eval loop.
# =============================================================================
def run_eval(
    policy: Policy,
    n_episodes: int = EVAL_EPISODES,
    seed_offset: int = 10000,
    verbose: bool = True,
    tb_logger: Optional["TBLogger"] = None,
) -> Dict[str, float]:
    env = build_environment()
    solver = setup_solver(env, policy)

    done_rates: List[float] = []
    deadlock_rates: List[float] = []
    rewards: List[float] = []
    steps_list: List[int] = []

    print(f"\n[Eval] {policy.get_name()}  x  {n_episodes} episodes")
    print("-" * 60)
    t0 = time.perf_counter()

    for ep in range(n_episodes):
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

        if tb_logger is not None:
            tb_logger.log_eval_episode(
                step=ep,
                done_rate=float(np.mean(done_rates)),
                deadlock_rate=float(np.mean(deadlock_rates)),
                episode_len=float(np.mean(steps_list)),
                total_reward=float(np.mean(rewards)),
            )

        if verbose and (ep + 1) % 5 == 0:
            print(f"  ep {ep+1:3d}/{n_episodes}  "
                  f"done={float(np.mean(done_rates)):.3f}  "
                  f"deadlock={float(np.mean(deadlock_rates)):.3f}  "
                  f"len={float(np.mean(steps_list)):.0f}")

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

    return {
        "done_rate": mean_done,
        "done_std": std_done,
        "deadlock_rate": mean_dl,
        "episode_len": mean_len,
        "total_reward": mean_rew,
        "n_episodes": len(done_rates),
    }


# =============================================================================
# DLA recording wrapper for demo collection.
# =============================================================================
class DLARecordingWrapper(Policy):

    def __init__(self):
        super().__init__()
        self.dla = DLAWrapper()
        self.demos: List[Tuple] = []

    def get_name(self):
        return "DLARecordingWrapper"

    def reset(self, env):
        self.dla.reset(env)

    def start_step(self, train: bool):
        self.dla.start_step(train)

    def end_episode(self, train: bool):
        self.dla.end_episode(train)

    def act(self, handle, state, eps=0.0):
        action = int(self.dla.act(handle, state, eps))
        base_obs, opps, payload = MAPPOPolicy._unwrap_state(state)
        self.demos.append((
            base_obs[:MAPPOPolicy.BASE_DIM].copy(),
            [np.asarray(o, dtype=np.float32).flatten()[:MAPPOPolicy.BASE_DIM].copy() for o in opps],
            payload,
            action,
        ))
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

    print(f"\n[DemoCollect] DLA x {n_episodes} episodes ...")
    t0 = time.perf_counter()

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
        if verbose and (ep + 1) % 10 == 0:
            print(f"  ep {ep+1:3d}/{n_episodes}  demos collected: {len(recorder.demos)}")

    elapsed = time.perf_counter() - t0
    print(f"[DemoCollect] {len(recorder.demos)} demos in {elapsed:.1f}s")
    return recorder.demos


# =============================================================================
# BC pretraining mode.
# =============================================================================
def run_bc(n_demo_episodes: int = 200, n_bc_epochs: int = 10):
    print("\n" + "=" * 70)
    print("BEHAVIOR CLONING from DLA expert")
    print("=" * 70)
    bc_logger = TBLogger(run_name="bc")

    demos = collect_dla_demos(n_episodes=n_demo_episodes)
    if len(demos) == 0:
        print("[BC] No demos collected — aborting.")
        return

    policy = MAPPOPolicy(
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
    policy.save(BC_CHECKPOINT)

    print(f"\n[BC] Final  loss={stats['bc_loss']:.4f}  acc={stats['bc_acc']:.3f}")
    print(f"[BC] Checkpoint saved: {BC_CHECKPOINT}")

    bc_logger.log_bc_epoch(epoch=n_bc_epochs, loss=stats["bc_loss"], accuracy=stats["bc_acc"])

    print(f"\n[BC] Evaluating BC-pretrained policy on {EVAL_EPISODES} episodes ...")
    metrics = run_eval(policy, n_episodes=EVAL_EPISODES, verbose=False, tb_logger=bc_logger)
    print(f"[BC] BC-policy done_rate = {metrics['done_rate']:.3f}")
    bc_logger.close()


# =============================================================================
# MAPPO training mode.
# =============================================================================
def run_train(n_episodes: int = 2000):
    print("\n" + "=" * 70)
    print(f"MAPPO TRAINING  ({n_episodes} episodes)")
    print("=" * 70)

    env = build_environment()
    policy = MAPPOPolicy(
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

    if os.path.exists(BC_CHECKPOINT):
        print(f"[Train] Loading BC warmstart from {BC_CHECKPOINT}")
        policy.load(BC_CHECKPOINT)
        policy.episode_count = 0
    elif os.path.exists(MAPPO_CHECKPOINT):
        print(f"[Train] Resuming from {MAPPO_CHECKPOINT}")
        policy.load(MAPPO_CHECKPOINT)
    else:
        print("[Train] No warmstart — training from scratch.")

    solver = setup_solver(env, policy)
    train_logger = TBLogger(run_name="train_mappo")

    t0 = time.perf_counter()
    done_history: List[float] = []
    reward_history: List[float] = []
    eval_log: List[Dict[str, Any]] = []

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

        # Per-episode TensorBoard logging (using rolling 50-ep window).
        recent_done_now = float(np.mean(done_history[-50:]))
        recent_rew_now = float(np.mean(reward_history[-50:]))
        train_logger.log_train_episode(
            episode=ep,
            done_50=recent_done_now,
            reward_50=recent_rew_now,
            eps=eps,
            ppo_stats=policy.last_train_stats,
        )

        if (ep + 1) % 10 == 0:
            recent_done = float(np.mean(done_history[-50:]))
            recent_rew = float(np.mean(reward_history[-50:]))
            stats = policy.last_train_stats
            elapsed = time.perf_counter() - t0
            print(
                f"[Train] ep {ep+1:4d}/{n_episodes}  "
                f"done50={recent_done:.3f}  "
                f"rew50={recent_rew:+.1f}  "
                f"eps={eps:.3f}  "
                f"v={stats.get('v_loss', 0):.3f} "
                f"p={stats.get('p_loss', 0):+.4f} "
                f"H={stats.get('ent', 0):.3f} "
                f"KL={stats.get('kl', 0):.4f}  "
                f"({elapsed:.0f}s)"
            )

        if (ep + 1) % 100 == 0:
            print(f"\n[Train] Mid-training eval at episode {ep+1} ...")
            eval_metrics = run_eval(policy, n_episodes=10, verbose=False, tb_logger=train_logger)
            eval_log.append({"episode": ep + 1, **eval_metrics})
            policy.save(MAPPO_CHECKPOINT)

    policy.save(MAPPO_CHECKPOINT)
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
def run_eval_mode(policy_name: str, n_episodes: int = EVAL_EPISODES):
    print("\n" + "=" * 70)
    print(f"EVAL MODE  ({policy_name}, {n_episodes} episodes)")
    print("=" * 70)

    if policy_name == "random":
        policy = RandomPolicy()
    elif policy_name == "dla":
        policy = DLAWrapper()
    elif policy_name == "mappo":
        policy = MAPPOPolicy(hidden=MAPPO_HIDDEN, device=DEVICE)
        ckpt = MAPPO_CHECKPOINT if os.path.exists(MAPPO_CHECKPOINT) else BC_CHECKPOINT
        if os.path.exists(ckpt):
            policy.load(ckpt)
        else:
            print(f"[Eval] WARNING: No checkpoint at {MAPPO_CHECKPOINT} or {BC_CHECKPOINT}")
            print("[Eval] Evaluating random-init MAPPO (expect very low done-rate).")
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    logger = TBLogger(run_name=f"eval_{policy_name}")
    try:
        return run_eval(policy, n_episodes=n_episodes, tb_logger=logger)
    finally:
        logger.close()


# =============================================================================
# CLI.
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Flatland MARL — simple training/eval")
    parser.add_argument("--mode", choices=["eval", "bc", "train"], required=True)
    parser.add_argument("--policy", choices=["random", "dla", "mappo"], default="mappo")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--bc-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("\n" + "=" * 60)
    print(f"  Flatland MARL — mode={args.mode}")
    print("=" * 60)

    if args.mode == "eval":
        n_ep = args.episodes if args.episodes is not None else EVAL_EPISODES
        run_eval_mode(args.policy, n_episodes=n_ep)
    elif args.mode == "bc":
        n_demo_ep = args.episodes if args.episodes is not None else 200
        run_bc(n_demo_episodes=n_demo_ep, n_bc_epochs=args.bc_epochs)
    elif args.mode == "train":
        n_ep = args.episodes if args.episodes is not None else 2000
        run_train(n_episodes=n_ep)


if __name__ == "__main__":
    main()
