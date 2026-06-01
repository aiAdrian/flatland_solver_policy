"""
Full IL → PPO Pipeline Runner
=============================

Convenience wrapper around the three-stage pipeline:

    1. il_collect_demos.py   — DeadLockAvoidancePolicy generates demonstrations
    2. il_pretrain.py        — DeciderNetwork is trained via BC on the demos
    3. marl_attention_temporal.py
                             — PPO fine-tunes from the BC checkpoint

Each stage can be skipped via CLI flag if its artefact already exists.

Usage
-----
    python run_il_then_ppo.py              # run all three stages
    python run_il_then_ppo.py --skip-collect          # reuse il_demos.pkl
    python run_il_then_ppo.py --skip-collect --skip-bc # reuse both, only PPO

Implementation notes
--------------------
* We invoke the three scripts as subprocesses (rather than importing) because
  each holds its own configuration / global state and is designed to run as
  __main__. This keeps the modules independently testable.
* The PYTHONPATH is propagated from the calling environment so package
  imports (`flatland`, `policy`, ...) resolve correctly.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_FILE = os.path.join(HERE, 'il_demos.pkl')
BC_CHECKPOINT = os.path.join(HERE, 'il_bc_checkpoint.pt')


def _run(script: str, env_extra: dict | None = None) -> None:
    print("\n" + "#" * 80)
    print(f"# Running: {script}")
    print("#" * 80, flush=True)
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    cmd = [sys.executable, '-u', os.path.join(HERE, script)]
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=HERE, env=env)
    dt = time.perf_counter() - t0
    if rc != 0:
        raise RuntimeError(f"{script} failed with exit code {rc} after {dt:.1f}s")
    print(f"# {script} finished in {dt/60:.1f} min", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-collect', action='store_true',
                        help=f'Skip demo collection if {DEMO_FILE} exists.')
    parser.add_argument('--skip-bc', action='store_true',
                        help=f'Skip BC pre-train if {BC_CHECKPOINT} exists.')
    parser.add_argument('--skip-ppo', action='store_true',
                        help='Skip the PPO fine-tune (collect + BC only).')
    parser.add_argument('--pure-marl', action='store_true',
                        help='Run pure MARL PPO only (skip IL demo collection + BC).')
    args = parser.parse_args()

    # Pure-MARL mode can be enabled via CLI flag or env variable.
    pure_marl = args.pure_marl or \
        os.environ.get('FLATLAND_POLICY_MODE', '').strip().lower() == 'pure_marl'

    if pure_marl:
        print('[runner] Pure-MARL mode enabled: skipping IL collection + BC pretrain.')
        if args.skip_ppo:
            print('[runner] Stage 3 SKIPPED via --skip-ppo.')
            return
        _run('marl_attention_temporal.py', env_extra={'FLATLAND_POLICY_MODE': 'pure_marl'})
        return

    # ------------------------------------------------------------------
    # Stage 1: demonstration collection
    # ------------------------------------------------------------------
    if args.skip_collect and os.path.exists(DEMO_FILE):
        print(f"[runner] Stage 1 SKIPPED — {DEMO_FILE} already exists.")
    else:
        _run('il_collect_demos.py')

    # ------------------------------------------------------------------
    # Stage 2: behaviour cloning
    # ------------------------------------------------------------------
    if args.skip_bc and os.path.exists(BC_CHECKPOINT):
        print(f"[runner] Stage 2 SKIPPED — {BC_CHECKPOINT} already exists.")
    else:
        _run('il_pretrain.py')

    # ------------------------------------------------------------------
    # Stage 3: PPO fine-tune from BC checkpoint
    # ------------------------------------------------------------------
    if args.skip_ppo:
        print("[runner] Stage 3 SKIPPED via --skip-ppo.")
        return

    if not os.path.exists(BC_CHECKPOINT):
        raise RuntimeError(
            f"BC checkpoint missing: {BC_CHECKPOINT}. "
            f"Cannot start PPO fine-tune."
        )

    _run('marl_attention_temporal.py',
         env_extra={
             'IL_LOAD_CHECKPOINT': BC_CHECKPOINT,
             'FLATLAND_POLICY_MODE': 'decider',
         })


if __name__ == '__main__':
    main()
