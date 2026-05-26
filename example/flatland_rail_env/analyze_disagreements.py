"""
analyze_disagreements.py  (V2 — Wrapper-based)
==============================================

Run two episodes per map:
  - One with DLA (records all decisions)
  - One with MAPPO (records all decisions)

Then compare the action distributions per cell type.

Usage:
    python analyze_disagreements.py --episodes 10
"""

import argparse
import os
import sys
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'
)))

from train_marl import (
    build_environment,
    MAPPO_CHECKPOINT,
    MAX_EPISODE_STEPS,
    DLAWrapper,
)
from mappo_policy import MAPPOPolicy

try:
    from marl_attention_temporal_observation.decision_point_utils import (
        DecisionPointUtils,
    )
except ImportError:
    DecisionPointUtils = None
    print("[warn] DecisionPointUtils not importable.")


# ============================================================
# Action labels
# ============================================================
ACTION_NAMES = {
    0: "DO_NOTHING",
    1: "MOVE_LEFT",
    2: "MOVE_FORWARD",
    3: "MOVE_RIGHT",
    4: "STOP_MOVING",
}


# ============================================================
# Logging Wrapper around any Policy
# ============================================================
class LoggingPolicy:
    """Wraps a policy and records every (cell_type, action, obs) tuple."""

    def __init__(self, inner, name: str):
        self.inner = inner
        self.name = name
        self.log: List[Dict[str, Any]] = []
        self._raw_env = None

    def get_name(self) -> str:
        return self.inner.get_name()

    def reset(self, env):
        self.inner.reset(env)
        self._raw_env = env.raw_env if hasattr(env, 'raw_env') else env

    def start_step(self, train: bool):
        if hasattr(self.inner, 'start_step'):
            self.inner.start_step(train)

    def end_episode(self, train: bool):
        if hasattr(self.inner, 'end_episode'):
            self.inner.end_episode(train)

    def end_step(self, train: bool):
        if hasattr(self.inner, 'end_step'):
            self.inner.end_step(train)

    def act(self, handle, state, eps: float = 0.0):
        action = self.inner.act(handle, state, eps)

        # Log this action with context
        ct = self._classify(handle)
        obs_vec = self._extract_features(state)
        self.log.append({
            'agent': handle,
            'cell_type': ct,
            'action': int(action),
            'obs': obs_vec,
            'episode': len([e for e in self.log if e.get('episode') is None]),  # placeholder
        })
        return action

    def save(self, *args, **kwargs):
        if hasattr(self.inner, 'save'):
            self.inner.save(*args, **kwargs)

    def load(self, *args, **kwargs):
        if hasattr(self.inner, 'load'):
            self.inner.load(*args, **kwargs)

    def _classify(self, handle: int) -> str:
        if self._raw_env is None or DecisionPointUtils is None:
            return "UNKNOWN"
        try:
            agent = self._raw_env.agents[handle]
            return DecisionPointUtils.classify_cell_type(agent, self._raw_env)
        except Exception:
            return "UNKNOWN"

    def reset_log(self):
        self.log.clear()

    def _extract_features(self, state: Any) -> np.ndarray:
        try:
            s = state
            if isinstance(s, list) and len(s) > 0:
                s = s[-1]
            if isinstance(s, tuple) and len(s) > 0:
                s = s[0]
            arr = np.asarray(s).flatten()
            if arr.shape[0] >= 22:
                return arr[:22]
            return arr
        except Exception:
            return np.zeros(22)

    # White-list of methods/attributes we DON'T delegate (handled here)
    _OWN_ATTRS = {
        'inner', 'name', 'log', '_raw_env',
        'reset', 'reset_log', 'act',
        'start_step', 'end_step', 'end_episode', 'save', 'load',
        'get_name', '_classify', '_extract_features',
    }

    def __getattr__(self, name):
        """Delegate any unknown method/attribute to the inner policy."""
        if name.startswith('_') or name in self._OWN_ATTRS:
            raise AttributeError(name)
        # Use object.__getattribute__ to avoid recursion via self.inner
        try:
            inner = object.__getattribute__(self, 'inner')
        except AttributeError:
            raise AttributeError(name)
        return getattr(inner, name)



# ============================================================
# Run one episode with given policy and collect log
# ============================================================
def run_episode(env, logging_policy: LoggingPolicy, ep_idx: int) -> Dict[str, Any]:
    """Run a single episode using the solver, return aggregated log."""
    from solver.flatland.flatland_solver import FlatlandSolver

    logging_policy.reset(env)
    logging_policy.reset_log()

    solver = FlatlandSolver(env, logging_policy)
    if hasattr(solver, "set_max_steps"):
        solver.set_max_steps(MAX_EPISODE_STEPS)

    tot_reward, tot_terminate, tot_steps = solver.run_episode(
        episode=ep_idx,
        env=env,
        policy=logging_policy,
        eps=0.0,
        training_mode=False,
    )

    return {
        'reward': float(tot_reward),
        'done_rate': float(tot_terminate),
        'steps': int(tot_steps),
        'log': list(logging_policy.log),
    }


# ============================================================
# Compare two logs at the cell-type level
# ============================================================
def compare_logs(
    dla_log: List[Dict[str, Any]],
    mappo_log: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate action distributions per cell type for both policies."""

    def by_cell_type(log):
        d = defaultdict(Counter)
        for e in log:
            d[e['cell_type']][e['action']] += 1
        return d

    dla_dist = by_cell_type(dla_log)
    mappo_dist = by_cell_type(mappo_log)

    return {
        'dla_dist': dla_dist,
        'mappo_dist': mappo_dist,
    }


# ============================================================
# Reporting
# ============================================================
def print_report(all_results: List[Dict[str, Any]]) -> None:
    print()
    print("=" * 78)
    print("  DLA vs MAPPO — Action distribution comparison")
    print("=" * 78)

    # Aggregate over all episodes
    dla_total = defaultdict(Counter)
    mappo_total = defaultdict(Counter)

    dla_done = []
    mappo_done = []

    for r in all_results:
        for ct, c in r['comparison']['dla_dist'].items():
            dla_total[ct].update(c)
        for ct, c in r['comparison']['mappo_dist'].items():
            mappo_total[ct].update(c)
        dla_done.append(r['dla_episode']['done_rate'])
        mappo_done.append(r['mappo_episode']['done_rate'])

    print(f"\n  Episodes per policy: {len(all_results)}")
    print(f"  DLA   mean done_rate = {np.mean(dla_done):.3f}")
    print(f"  MAPPO mean done_rate = {np.mean(mappo_done):.3f}")
    print()

    all_cell_types = sorted(set(list(dla_total.keys()) + list(mappo_total.keys())))

    for ct in all_cell_types:
        dla_acts = dla_total.get(ct, Counter())
        mappo_acts = mappo_total.get(ct, Counter())

        dla_n = sum(dla_acts.values())
        mappo_n = sum(mappo_acts.values())

        if dla_n == 0 and mappo_n == 0:
            continue

        print("-" * 78)
        print(f"  Cell type: {ct}")
        print(f"    DLA   total decisions: {dla_n}")
        print(f"    MAPPO total decisions: {mappo_n}")
        print()
        print(f"    {'Action':<14s}  {'DLA':>10s}  {'MAPPO':>10s}  {'diff':>10s}")
        all_acts = sorted(set(list(dla_acts.keys()) + list(mappo_acts.keys())))
        for a in all_acts:
            dla_pct = dla_acts.get(a, 0) / max(dla_n, 1)
            mappo_pct = mappo_acts.get(a, 0) / max(mappo_n, 1)
            diff = mappo_pct - dla_pct
            marker = ""
            if abs(diff) > 0.10:
                marker = "  ⚠️ "
            print(f"    {ACTION_NAMES.get(a, str(a)):<14s}  "
                  f"{dla_pct:>9.1%}  {mappo_pct:>9.1%}  "
                  f"{diff:>+9.1%}{marker}")
    print()
    print("=" * 78)
    print("  Hypotheses:")
    print("=" * 78)

    # Specific checks at decision-point cell types
    for ct in ("PRE_M", "MERGING", "SWITCH"):
        if ct not in dla_total and ct not in mappo_total:
            continue
        dla_acts = dla_total.get(ct, Counter())
        mappo_acts = mappo_total.get(ct, Counter())
        dla_n = max(sum(dla_acts.values()), 1)
        mappo_n = max(sum(mappo_acts.values()), 1)

        dla_stop = dla_acts.get(4, 0) / dla_n
        mappo_stop = mappo_acts.get(4, 0) / mappo_n
        dla_fwd = dla_acts.get(2, 0) / dla_n
        mappo_fwd = mappo_acts.get(2, 0) / mappo_n

        if mappo_stop > dla_stop + 0.15:
            print(f"  🟠 At {ct}: MAPPO STOPs much more than DLA "
                  f"({mappo_stop:.0%} vs {dla_stop:.0%}) — overly cautious.")
        if mappo_fwd > dla_fwd + 0.15:
            print(f"  🔴 At {ct}: MAPPO goes FORWARD much more than DLA "
                  f"({mappo_fwd:.0%} vs {dla_fwd:.0%}) — underestimates conflict.")

    print("=" * 78)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default=MAPPO_CHECKPOINT)
    args = parser.parse_args()

    print()
    print("=" * 78)
    print(f"  Disagreement Analysis — {args.episodes} episodes per policy")
    print("=" * 78)

    # Build env and policies (we need fresh envs for each pair of episodes)
    env_dla = build_environment()
    env_mappo = build_environment()

    dla = DLAWrapper()
    dla_logger = LoggingPolicy(dla, "DLA")

    # Pick up the right BASE_DIM matching our observation config
    from train_marl import _get_observation_base_dim, BC_CHECKPOINT
    base_dim = _get_observation_base_dim()
    print(f">> Building MAPPO with base_dim={base_dim}")

    mappo = MAPPOPolicy(base_dim=base_dim, device='cpu')
    mappo.reset(env_mappo)

    # Try MAPPO checkpoint first, fall back to BC checkpoint
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        if os.path.exists(BC_CHECKPOINT):
            print(f"[info] No MAPPO checkpoint at {ckpt_path}")
            print(f"[info] Falling back to BC checkpoint: {BC_CHECKPOINT}")
            ckpt_path = BC_CHECKPOINT
        else:
            print(f"[error] No checkpoint found (tried {ckpt_path} and {BC_CHECKPOINT})")
            sys.exit(1)
    mappo.load(ckpt_path)
    mappo_logger = LoggingPolicy(mappo, "MAPPO")

    # Run pairs of episodes
    all_results = []
    for ep in range(args.episodes):
        print(f"\n  Episode {ep+1}/{args.episodes}")

        print(f"    Running DLA ...", end=' ', flush=True)
        try:
            dla_ep = run_episode(env_dla, dla_logger, ep_idx=ep)
            print(f"done={dla_ep['done_rate']:.0%}, steps={dla_ep['steps']}, "
                  f"decisions={len(dla_ep['log'])}")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            continue

        print(f"    Running MAPPO ...", end=' ', flush=True)
        try:
            mappo_ep = run_episode(env_mappo, mappo_logger, ep_idx=ep)
            print(f"done={mappo_ep['done_rate']:.0%}, steps={mappo_ep['steps']}, "
                  f"decisions={len(mappo_ep['log'])}")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            continue

        comparison = compare_logs(dla_ep['log'], mappo_ep['log'])

        all_results.append({
            'dla_episode': dla_ep,
            'mappo_episode': mappo_ep,
            'comparison': comparison,
        })

    if not all_results:
        print("\n[error] No successful episode pairs.")
        sys.exit(1)

    print_report(all_results)


if __name__ == "__main__":
    main()

