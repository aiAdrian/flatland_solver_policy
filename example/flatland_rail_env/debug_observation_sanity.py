"""
debug_observation_sanity.py
===========================
Sammelt Observations über N Episoden und prüft auf:
  - Werte ausserhalb [0, 1] (falsche Normierung)
  - Duplikat-Features (perfekte Korrelation)
  - Tote Features (Varianz ≈ 0)
  - Feature-[6]-Dual-Bedeutung (Rank vs Progress-Gain)
  - decision_type Verteilung

Ausführen:
    cd flatland_solver_policy
    python example/flatland_rail_env/debug_observation_sanity.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rail_env import RailEnv

from marl_attention_temporal_observation.decision_point_observation import DecisionPointObservation

# ── Konfiguration ────────────────────────────────────────────────────────────
N_EPISODES   = 30
MAX_STEPS    = 200
N_AGENTS     = 5
WIDTH        = 30
HEIGHT       = 30
OBS_SIZE     = DecisionPointObservation.OBS_SIZE
RANGE_TOL    = 0.05   # erlaubte Abweichung von [0,1]

# ── Environment aufbauen ─────────────────────────────────────────────────────
def make_env():
    obs_builder = DecisionPointObservation(
        observation_profile="local_tree_encoder",
        use_trainable_tree_encoder=True,
    )
    env = RailEnv(
        width=WIDTH,
        height=HEIGHT,
        rail_generator=sparse_rail_generator(max_num_cities=4, seed=42),
        line_generator=sparse_line_generator(seed=42),
        number_of_agents=N_AGENTS,
        obs_builder_object=obs_builder,
    )
    return env, obs_builder


# ── Daten sammeln ────────────────────────────────────────────────────────────
def collect_observations():
    env, obs_builder = make_env()
    all_obs = []      # alle rohen Feature-Vektoren
    f6_at_switch = []
    f6_no_switch  = []

    for ep in range(N_EPISODES):
        obs_list, _ = env.reset(regenerate_rail=True, regenerate_schedule=True)
        done = {'__all__': False}
        step = 0
        while not done['__all__'] and step < MAX_STEPS:
            step += 1
            actions = {h: 2 for h in range(N_AGENTS)}   # alle immer Forward
            obs_list, _, done, _ = env.step(actions)
            for h in range(N_AGENTS):
                pair = obs_list[h]
                if pair is None:
                    continue
                # get() gibt (features, opp_handles) zurück
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    feat = np.asarray(pair[0], dtype=np.float32)
                else:
                    feat = np.asarray(pair, dtype=np.float32)
                if feat.shape[0] != OBS_SIZE:
                    continue
                all_obs.append(feat.copy())

                # Feature[6] Dual-Bedeutung tracken
                decision_raw = feat[0] * 8.0   # back-encode
                at_switch = (round(decision_raw) & 2) != 0
                if at_switch:
                    f6_at_switch.append(feat[6])
                else:
                    f6_no_switch.append(feat[6])

        if (ep + 1) % 5 == 0:
            print(f"  Episode {ep+1}/{N_EPISODES} gesammelt, Beobachtungen bisher: {len(all_obs)}")

    return np.array(all_obs, dtype=np.float32), f6_at_switch, f6_no_switch


# ── Analyse ──────────────────────────────────────────────────────────────────
def run_analysis():
    print("=" * 65)
    print("Observation Sanity Check")
    print("=" * 65)

    print(f"\nSammle Beobachtungen ({N_EPISODES} Episoden, {N_AGENTS} Agenten) …")
    data, f6_switch, f6_no_switch = collect_observations()
    N = data.shape[0]
    print(f"  → {N} Observations à {OBS_SIZE}D gesammelt.\n")

    if N < 10:
        print("FEHLER: Zu wenige Beobachtungen – Environment konfiguration prüfen.")
        return

    # 1) Bereichs-Check ──────────────────────────────────────────────────────
    print("─── 1) Werte ausserhalb [0, 1] ───────────────────────────────────")
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    out_of_range = []
    for i in range(OBS_SIZE):
        lo, hi = mins[i], maxs[i]
        if lo < -RANGE_TOL or hi > 1.0 + RANGE_TOL:
            out_of_range.append((i, lo, hi))

    if out_of_range:
        print(f"  ⚠  {len(out_of_range)} Feature(s) AUSSERHALB [0, 1]:")
        for idx, lo, hi in out_of_range:
            print(f"     Feature[{idx:2d}]  min={lo:.4f}  max={hi:.4f}")
    else:
        print("  ✓  Alle Features in [0, 1]")

    # 2) Tote Features ────────────────────────────────────────────────────────
    print("\n─── 2) Tote Features (std ≈ 0) ───────────────────────────────────")
    stds = data.std(axis=0)
    dead = [(i, stds[i]) for i in range(OBS_SIZE) if stds[i] < 1e-4]
    if dead:
        print(f"  ⚠  {len(dead)} Feature(s) praktisch konstant (std < 1e-4):")
        for idx, s in dead:
            print(f"     Feature[{idx:2d}]  mean={data[:,idx].mean():.4f}  std={s:.6f}")
    else:
        print("  ✓  Kein Feature konstant")

    # 3) Duplikat-Features (Korrelation ≈ ±1) ──────────────────────────────
    print("\n─── 3) Duplikat-Features (|corr| > 0.99) ─────────────────────────")
    # Nur Features mit std > 0 einbeziehen
    valid = [i for i in range(OBS_SIZE) if stds[i] > 1e-6]
    corr  = np.corrcoef(data[:, valid].T)
    duplicates = []
    for ii in range(len(valid)):
        for jj in range(ii + 1, len(valid)):
            c = corr[ii, jj]
            if abs(c) > 0.99:
                duplicates.append((valid[ii], valid[jj], c))
    if duplicates:
        print(f"  ⚠  {len(duplicates)} Duplikat-Paar(e):")
        for a, b, c in duplicates:
            print(f"     Feature[{a:2d}] ↔ Feature[{b:2d}]  corr={c:.5f}")
    else:
        print("  ✓  Keine perfekten Duplikate")

    # 4) Feature[5] vs Feature[53] ──────────────────────────────────────────
    print("\n─── 4) Feature[5] vs Feature[53] (erwartetes Duplikat) ───────────")
    diff_5_53 = np.abs(data[:, 5] - data[:, 53])
    print(f"  |feat[5] - feat[53]|  max={diff_5_53.max():.6f}  mean={diff_5_53.mean():.6f}")
    if diff_5_53.max() < 1e-6:
        print("  ⚠  Feature[53] ist exakt identisch mit Feature[5] → verschwendete Dimension")
    else:
        print("  ✓  Unterschiedlich (unerwartet – bitte prüfen)")

    # 5) Feature[6] Dual-Bedeutung ─────────────────────────────────────────
    print("\n─── 5) Feature[6] Dual-Bedeutung (Rank vs Progress-Gain) ─────────")
    if f6_switch and f6_no_switch:
        arr_sw  = np.array(f6_switch)
        arr_nsw = np.array(f6_no_switch)
        print(f"  Bei Switch     : n={len(arr_sw)}  mean={arr_sw.mean():.4f}  std={arr_sw.std():.4f}  "
              f"min={arr_sw.min():.4f}  max={arr_sw.max():.4f}")
        print(f"  Kein Switch    : n={len(arr_nsw)}  mean={arr_nsw.mean():.4f}  std={arr_nsw.std():.4f}  "
              f"min={arr_nsw.min():.4f}  max={arr_nsw.max():.4f}")
        # Welch t-Test (einfach manuell)
        diff_means = abs(arr_sw.mean() - arr_nsw.mean())
        pooled_std = (arr_sw.std() + arr_nsw.std()) / 2.0
        if pooled_std > 1e-6:
            print(f"  Mittlere Abweichung der Verteilungen: {diff_means:.4f}  "
                  f"(relativ zu Streu: {diff_means/pooled_std:.2f}×σ)")
        if diff_means > 0.05:
            print("  ⚠  Feature[6] hat am Switch vs. kein-Switch unterschiedliche Statistik")
            print("     → Netz sieht zwei semantisch verschiedene Bedeutungen an derselben Position")
    else:
        print("  (nicht genug switch-Daten gesammelt)")

    # 6) decision_type Verteilung ─────────────────────────────────────────
    print("\n─── 6) decision_type Verteilung (Feature[0]) ──────────────────────")
    raw_dt = (data[:, 0] * 8.0).round().astype(int)
    for dt_val in sorted(set(raw_dt)):
        cnt = (raw_dt == dt_val).sum()
        label = {0: "keine", 1: "READY_TO_DEPART", 2: "switch", 4: "merge",
                 6: "switch+merge", 8: "DONE"}.get(dt_val, f"?({dt_val})")
        print(f"  dt={dt_val} ({label:20s})  {cnt:5d}  ({100*cnt/N:.1f}%)")

    # 7) Per-Feature Statistik ──────────────────────────────────────────────
    print("\n─── 7) Per-Feature Statistik (nur aktive Features) ───────────────")
    print(f"  {'Feat':>5}  {'mean':>7}  {'std':>7}  {'min':>7}  {'max':>7}")
    for i in range(OBS_SIZE):
        m, s, lo, hi = data[:,i].mean(), stds[i], mins[i], maxs[i]
        flag = ""
        if lo < -RANGE_TOL or hi > 1.0 + RANGE_TOL:
            flag += " OUT_OF_RANGE"
        if s < 1e-4:
            flag += " DEAD"
        print(f"  [{i:2d}]   {m:7.4f}  {s:7.4f}  {lo:7.4f}  {hi:7.4f}{flag}")

    print("\n" + "=" * 65)
    print("Fertig.")


if __name__ == "__main__":
    run_analysis()
