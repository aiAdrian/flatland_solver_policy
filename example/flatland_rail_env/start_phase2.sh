#!/bin/bash
# ============================================================
# Phase 2: Längeres Training mit demselben Setup
# ============================================================
set -e

cd /home/u216993/workspace/ai4realnet/aiAdrian_flatland/flatland_solver_policy/example/flatland_rail_env

echo "=========================================="
echo "Phase 2 — 300 episodes, same setup"
echo "Start: $(date)"
echo "=========================================="

# Sichere aktuellen BEST checkpoint nochmal explizit
cp mappo_FINAL_done75_20260528.pt mappo_phase2_start.pt
echo "✓ Start checkpoint: mappo_phase2_start.pt"

# Kopiere ihn als 'aktuelles' MAPPO ckpt (warmstart)
cp mappo_FINAL_done75_20260528.pt mappo_checkpoint_conflict_aware.pt
echo "✓ Loaded as warmstart"

# Run training (300 episoden, mehr Zeit zu lernen)
python train_marl.py \
    --mode train \
    --episodes 300 \
    --obs conflict_aware \
    2>&1 | tee phase2_run_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "Phase 2 done: $(date)"
echo "=========================================="

# Backup result
cp mappo_checkpoint_conflict_aware.pt mappo_PHASE2_FINAL_$(date +%Y%m%d).pt
ls -la mappo_*.pt

