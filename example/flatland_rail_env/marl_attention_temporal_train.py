"""
MARL Attention Temporal — Full Training Launcher
================================================

Single entry-point that:

    1. Asks the user (interactively) whether to clean the prior artefacts
       (`runs/`, `training_output/`, `generated_envs*/`, `il_demos.pkl`,
       `il_bc_checkpoint.pt`, `il_pipeline.log`, `training.log`).
    2. Sets PYTHONPATH so the package imports resolve correctly without the
       user having to remember the long shell incantation.
    3. Runs the three-stage IL → PPO pipeline by invoking
       `run_il_then_ppo.py` as a subprocess.

Usage
-----
    python marl_attention_temporal_train.py
        # interactive prompts for cleanup, then runs full pipeline

    python marl_attention_temporal_train.py --yes
        # accept ALL cleanup prompts (delete everything before training)

    python marl_attention_temporal_train.py --no
        # keep ALL existing artefacts (resume / reuse demos & BC checkpoint)

    python marl_attention_temporal_train.py --skip-collect --skip-bc
        # forwarded to run_il_then_ppo.py — only re-run PPO

Design notes
------------
* The cleanup targets cover every place the pipeline writes:
    runs/                       TensorBoard event files (base_solver.py)
    training_output/            saved policy snapshots (base_solver.py)
    generated_envs/             RailEnv pickles for PURE_MARL eval set
    generated_envs_il/          RailEnv pickles for IL demo collection
    generated_envs_<phase>*/    per-phase curriculum env caches
    il_demos.pkl                Stage 1 output
    il_bc_checkpoint.pt         Stage 2 output
    il_pipeline.log             prior pipeline log
    training.log                prior training log
* We deliberately do NOT delete `decider_policy.py.last_*` or anything that
  isn't a generated artefact.
* Cleanup uses `shutil.rmtree(..., ignore_errors=True)` and `os.remove`
  with existence checks — never raises if a path is already absent.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# Repo root computed relative to this file.
EXAMPLE_DIR = HERE
RAIL_ENV_DIR = os.path.dirname(EXAMPLE_DIR)              # .../example
SOLVER_POLICY_DIR = os.path.dirname(RAIL_ENV_DIR)        # .../flatland_solver_policy
WORKSPACE_ROOT = os.path.dirname(SOLVER_POLICY_DIR)      # .../aiAdrian_flatland
FLATLAND_RL_DIR = os.path.join(WORKSPACE_ROOT, 'flatland-rl')

PYTHONPATH_ENTRIES = [
    FLATLAND_RL_DIR,
    SOLVER_POLICY_DIR,
    EXAMPLE_DIR,
]


# ----------------------------------------------------------------------------
# Cleanup target groups. Each group has a label, a question, and a list of
# absolute paths. Directories and files mixed are fine -- _wipe handles both.
# ----------------------------------------------------------------------------
def _expand_glob(patterns):
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))
    return paths


def _cleanup_groups():
    return [
        {
            'label': 'TensorBoard logs',
            'question': "Delete TensorBoard `runs/` directory?",
            'paths': [os.path.join(EXAMPLE_DIR, 'runs')],
        },
        {
            'label': 'Saved policy snapshots',
            'question': "Delete `training_output/` (saved policy snapshots)?",
            'paths': [os.path.join(EXAMPLE_DIR, 'training_output')],
        },
        {
            'label': 'Generated environment caches',
            'question': "Delete generated environment caches "
                        "(`generated_envs/`, `generated_envs_il/`, `generated_envs/phase*`)?",
            'paths': _expand_glob([
                os.path.join(EXAMPLE_DIR, 'generated_envs'),
                os.path.join(EXAMPLE_DIR, 'generated_envs_il'),
                os.path.join(EXAMPLE_DIR, 'generated_envs', 'phase*'),
            ]),
        },
        {
            'label': 'IL demonstrations + BC checkpoint',
            'question': "Delete IL artefacts (`il_demos.pkl`, `il_bc_checkpoint.pt`)?",
            'paths': [
                os.path.join(EXAMPLE_DIR, 'il_demos.pkl'),
                os.path.join(EXAMPLE_DIR, 'il_bc_checkpoint.pt'),
            ],
        },
        {
            'label': 'Old log files',
            'question': "Delete prior log files (`il_pipeline.log`, `training.log`)?",
            'paths': [
                os.path.join(EXAMPLE_DIR, 'il_pipeline.log'),
                os.path.join(EXAMPLE_DIR, 'training.log'),
            ],
        },
    ]


def _path_summary(paths):
    """Return a short human-readable description of which paths exist."""
    rows = []
    for p in paths:
        if os.path.isdir(p):
            rows.append(f"  [dir]  {p}")
        elif os.path.isfile(p):
            try:
                size_mb = os.path.getsize(p) / (1024 * 1024)
                rows.append(f"  [file] {p}  ({size_mb:.2f} MB)")
            except OSError:
                rows.append(f"  [file] {p}")
        else:
            rows.append(f"  [---]  {p}  (does not exist)")
    return "\n".join(rows)


def _wipe(paths):
    deleted = 0
    for p in paths:
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
                deleted += 1
            elif os.path.isfile(p):
                os.remove(p)
                deleted += 1
        except OSError as exc:
            print(f"  WARN: could not delete {p}: {exc}")
    return deleted


def _read_single_key() -> str:
    """Read ONE keystroke from the terminal without requiring Enter.

    Falls back to line-based `input()` if the terminal is not a TTY (e.g.
    when stdin is redirected from a file or piped). Returns the lower-case
    character or an empty string on EOF.
    """
    # Pipe / non-interactive: fall back to line input.
    if not sys.stdin.isatty():
        try:
            return input().strip().lower()
        except EOFError:
            return ''

    # POSIX raw-mode single character (Linux/macOS).
    try:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        # Echo the key + newline so the user sees what was pressed.
        if ch in ('\r', '\n'):
            print()
            return ''
        if ch == '\x03':            # Ctrl-C
            raise KeyboardInterrupt
        print(ch)
        return ch.lower()
    except (ImportError, AttributeError, OSError):
        # Last resort: line input.
        try:
            return input().strip().lower()
        except EOFError:
            return ''


def _ask(question: str, default_yes: bool, force_yes: bool, force_no: bool) -> bool:
    """Return True if the user wants the action performed."""
    if force_yes:
        print(f"{question}  [auto: YES]")
        return True
    if force_no:
        print(f"{question}  [auto: no]")
        return False
    suffix = " [Y/n] " if default_yes else " [y/N] "
    while True:
        print(f"{question}{suffix}", end='', flush=True)
        ch = _read_single_key()
        if ch == '' and default_yes:
            return True
        if ch == '' and not default_yes:
            return False
        if ch in ('y', 'j'):
            return True
        if ch in ('n',):
            return False
        print("  Bitte 'y' oder 'n' druecken (Enter = Default).")


def _run_cleanup(force_yes: bool, force_no: bool) -> None:
    print("\n" + "=" * 80)
    print("CLEANUP — alte Trainingsartefakte")
    print("=" * 80)
    for group in _cleanup_groups():
        existing = [p for p in group['paths']
                    if os.path.isdir(p) or os.path.isfile(p)]
        if not existing:
            print(f"\n{group['label']}: nichts vorhanden, überspringe.")
            continue

        print(f"\n{group['label']}:")
        print(_path_summary(existing))
        if _ask(group['question'], default_yes=True,
                force_yes=force_yes, force_no=force_no):
            n = _wipe(existing)
            print(f"  → {n} Eintrag/Einträge gelöscht.")
        else:
            print("  → behalten.")


def _run_pipeline(extra_args: list[str]) -> int:
    print("\n" + "=" * 80)
    print("PIPELINE — IL → BC → PPO (run_il_then_ppo.py)")
    print("=" * 80)
    env = os.environ.copy()
    # Prepend our entries to PYTHONPATH (preserve any pre-existing value).
    existing = env.get('PYTHONPATH', '')
    parts = PYTHONPATH_ENTRIES + ([existing] if existing else [])
    env['PYTHONPATH'] = os.pathsep.join(p for p in parts if p)

    cmd = [sys.executable, '-u',
           os.path.join(EXAMPLE_DIR, 'run_il_then_ppo.py')] + extra_args
    print("Working dir   :", EXAMPLE_DIR)
    print("Python        :", sys.executable)
    print("PYTHONPATH    :", env['PYTHONPATH'])
    print("Command       :", " ".join(cmd))
    print("-" * 80, flush=True)

    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=EXAMPLE_DIR, env=env)
    dt = time.perf_counter() - t0
    print(f"\nPipeline finished in {dt/60:.1f} min with exit code {rc}.")
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full IL→PPO training launcher with optional cleanup.")
    parser.add_argument('--yes', action='store_true',
                        help='Auto-accept every cleanup prompt.')
    parser.add_argument('--no', action='store_true',
                        help='Auto-decline every cleanup prompt.')
    parser.add_argument('--skip-cleanup', action='store_true',
                        help='Skip cleanup phase entirely.')
    parser.add_argument('--skip-collect', action='store_true',
                        help='Forwarded to run_il_then_ppo.py (reuse il_demos.pkl).')
    parser.add_argument('--skip-bc', action='store_true',
                        help='Forwarded to run_il_then_ppo.py (reuse il_bc_checkpoint.pt).')
    parser.add_argument('--skip-ppo', action='store_true',
                        help='Forwarded to run_il_then_ppo.py (collect + BC only).')
    parser.add_argument('--pure-marl', action='store_true',
                        help='Forwarded to run_il_then_ppo.py (skip IL+BC; PPO only).')
    args = parser.parse_args()

    if args.yes and args.no:
        print("ERROR: --yes and --no are mutually exclusive.", file=sys.stderr)
        return 2

    if not args.skip_cleanup:
        _run_cleanup(force_yes=args.yes, force_no=args.no)
    else:
        print("[launcher] Skipping cleanup (--skip-cleanup).")

    extra = []
    if args.skip_collect:
        extra.append('--skip-collect')
    if args.skip_bc:
        extra.append('--skip-bc')
    if args.skip_ppo:
        extra.append('--skip-ppo')
    if args.pure_marl:
        extra.append('--pure-marl')

    return _run_pipeline(extra)


if __name__ == '__main__':
    raise SystemExit(main())
