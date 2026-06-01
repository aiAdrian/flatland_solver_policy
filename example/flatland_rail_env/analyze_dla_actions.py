"""Analyze DLA action distribution at decision points."""
import os, sys, numpy as np
from collections import Counter, defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for _p in (_PROJECT_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_marl import (
    build_environment, setup_solver, DLAWrapper,
    _set_selected_observation, _print_observation_info,
)
from marl_attention_temporal_observation.decision_point_utils import DecisionPointUtils
from flatland.envs.step_utils.states import TrainState

ACTION_NAMES = {0: "DO_NOTHING", 1: "MOVE_LEFT", 2: "MOVE_FORWARD",
                3: "MOVE_RIGHT", 4: "STOP_MOVING"}

class DLAActionAnalyzer(DLAWrapper):
    def __init__(self):
        super().__init__()
        self.records = []

    def act(self, handle, state, eps=0.0):
        action = super().act(handle, state, eps)
        try:
            agent = self.env.raw_env.agents[handle]
            if agent.state == TrainState.DONE:
                cell_type = "DONE"
            elif agent.position is None:
                cell_type = "OUTSIDE"
            else:
                cell_type = DecisionPointUtils.classify_cell_type(agent, self.env.raw_env)
        except Exception:
            cell_type = "UNKNOWN"
        self.records.append((cell_type, int(action)))
        return action


def main():
    _set_selected_observation("conflict_aware")
    _print_observation_info()
    n_episodes = 50
    print(f"\n[DLA-Analysis] Running DLA on {n_episodes} episodes ...")
    env = build_environment()
    analyzer = DLAActionAnalyzer()
    solver = setup_solver(env, analyzer)
    for ep in range(n_episodes):
        np.random.seed(30000 + ep)
        solver.run_episode(episode=ep, env=env, policy=analyzer, eps=0.0, training_mode=False)

    table = defaultdict(Counter)
    for cell_type, action in analyzer.records:
        table[cell_type][action] += 1

    cell_order = ["SWITCH", "MERGING", "PRE_M", "FORWARD_ONLY", "OUTSIDE", "DONE"]
    action_order = [0, 1, 2, 3, 4]

    print("\n" + "=" * 90)
    print(f"DLA ACTION DISTRIBUTION  ({n_episodes} ep, {len(analyzer.records)} samples)")
    print("=" * 90)
    header = f"{'CELL_TYPE':<15}"
    for a in action_order:
        header += f"{ACTION_NAMES[a]:>15}"
    header += f"{'TOTAL':>10}"
    print(header)
    print("-" * len(header))
    for ct in cell_order:
        if ct not in table:
            continue
        row = f"{ct:<15}"
        total_ct = sum(table[ct].values())
        for a in action_order:
            n = table[ct].get(a, 0)
            pct = 100.0 * n / max(1, total_ct)
            row += f"{n:>7} ({pct:>5.1f}%)"
        row += f"{total_ct:>10}"
        print(row)

    print("\n" + "=" * 90)
    print("DECISION POINTS ONLY (SWITCH + MERGING + PRE_M)")
    print("=" * 90)
    dp_records = [(c, a) for c, a in analyzer.records if c in ("SWITCH", "MERGING", "PRE_M")]
    dp_counter = Counter(a for _, a in dp_records)
    n_dp = len(dp_records)
    print(f"Total decision-point actions: {n_dp}\n")
    for a in action_order:
        n = dp_counter.get(a, 0)
        pct = 100.0 * n / max(1, n_dp)
        bar = "#" * int(60 * pct / 100)
        print(f"  {ACTION_NAMES[a]:<14}: {n:>5} ({pct:>5.2f}%)  {bar}")


if __name__ == "__main__":
    main()
