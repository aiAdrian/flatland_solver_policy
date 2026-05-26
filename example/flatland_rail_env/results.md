# MAPPO Diagnostic Report — 20 Episodes

## Performance
- DLA:    1.000 done_rate
- MAPPO:  0.610 done_rate

## Critical Findings

### Finding 1: No Spawn Throttling
MAPPO spawns 100% of the time at OUTSIDE,
DLA only 39% (61% wait). Causes city congestion.

### Finding 2: No Conflict Detection at MERGING
MAPPO 100% FORWARD at merging,
DLA 27% STOP. Causes head-on collisions.

### Finding 3: FORWARD Bias at SWITCH
MAPPO 82% FORWARD vs DLA 63%.
Lateral moves halved.

## Root Cause: Observation Lacks Coordination Info

The current 22-dim observation has no signal for:
- Global traffic density
- Other agents' next-step intentions
- Spawn coordination


python analyze_disagreements.py --episodes 20
python train_marl.py --mode train --episodes 200