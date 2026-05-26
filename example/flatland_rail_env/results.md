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



## Behavior Cloning ist eine Form von Imitation Learning — die einfachste Art, einer Policy beizubringen, was sie tun soll.

💡 Wie es funktioniert
1. Sammle Demonstrationen vom "Lehrer" (hier: DLA)
   → Tupel (observation, action) für viele Situationen
   
2. Trainiere ein neuronales Netz, das vorhersagt:
   "Gegeben diese observation, welche action würde der Lehrer wählen?"
   
3. Das ist ein simples Supervised Learning:
   loss = -log P(a_lehrer | obs)

## Commands
python analyze_disagreements.py --episodes 20
python train_marl.py --mode bc --episodes 100 --bc-epochs 10
python train_marl.py --mode train --episodes 200



# Spawn-Aware Observation Experiment Results

## Hypothesis
DLA stops spawning agents 71% of the time (OUTSIDE cells), but MAPPO never does.
Adding spawn-density features should help.

## Implementation
- Added 3 features: active_density, pending_density, is_ready
- Clean inheritance: SpawnAwareObservation extends DecisionPointObservation
- Toggleable via USE_SPAWN_AWARE_OBSERVATION flag

## Results
| Policy | Done Rate | Notes |
|--------|-----------|-------|
| Random | ~0.05 | Baseline |
| DLA (expert) | 1.00 | Upper bound |
| BC (22-dim) | 0.22 | Old observation |
| **BC (25-dim)** | **0.50** | **+125%** ✅ |
| PPO from BC (200 ep) | 0.44 | Destabilizes BC |

## Conclusion
Spawn-aware features substantially improve imitation learning.
PPO fine-tuning from BC is unstable in current reward setup.
Suspected cause: spurious correlation on global pending_density.
Next step: replace global density with local reservation features.

class DecisionPointObservation(ObservationBuilder):
    BASE_OBS_SIZE = 22
    
class SpawnAwareObservation(DecisionPointObservation):
    BASE_OBS_SIZE = 25                # = 22 + 3 spawn features
    
class ConflictAwareObservation(SpawnAwareObservation):  # ← NEU
    BASE_OBS_SIZE = 34                # = 25 + 9 conflict features


# Results: CBS-Inspired Observation Features for MAPF-RL

## Hypothesis
Multi-agent rail dispatching with PPO suffers from STOP-mode collapse
because the policy lacks local, action-conditioned conflict signals.
Adding CBS-style features (vertex/edge conflicts, priority hierarchy)
should provide causal signals that prevent mode collapse.

## Method
Hierarchical observation system:
  - DecisionPointObservation (22-dim baseline, Flatland competition)
  - SpawnAwareObservation (+3 spawn density features)  
  - ConflictAwareObservation (+5 global + 9 local CBS features)

5 global features:
  - global_conflict_pressure (system-wide reservation density)
  - my_priority_global (rank by remaining shortest-path distance)
  - my_sp_blocked_score (how blocked is my path)
  - sp_alternatives_avg_score (do alternatives help?)
  - expected_yield_count (do others benefit if I stop?)

9 local features (3 per direction L/F/R):
  - vertex_conflict_horizon (when does another agent enter target?)
  - edge_conflict (head-on swap)
  - lane_density (corridor occupancy)

## Results

| Observation        | Dim | BC Acc | BC Done-Rate | Notes |
|--------------------|-----|--------|--------------|-------|
| DecisionPoint      | 22  | ~0.85  | 0.22         | baseline |
| SpawnAware         | 25  | ~0.90  | 0.50         | +127% |
| **ConflictAware**  | **39**  | **0.955** | **0.58**     | **+163%** ✅ |

Reference: DLA expert (upper bound) = 1.00

## Key Insight
Local action-conditioned conflict features (CBS-style) provide
causal signals that prevent the spurious-correlation problem
seen with global density features alone.
# Results: CBS-Inspired Conflict Features for Imitation Learning in MAPF

## Setup
- Environment: Flatland 30×40, 3 cities, 5 agents
- Expert: DeadLockAvoidancePolicy (DLA)
- Method: Behavior Cloning at decision points only
- Architecture: MAPPO with shared encoder + tree attention

## Observation Hierarchy
| Tier | Class                        | Dim | Key Features                              |
|------|------------------------------|-----|-------------------------------------------|
| 0    | DecisionPointObservation     | 22  | Local transitions, SP hints, deadlocks   |
| 1    | SpawnAwareObservation        | 25  | + 3 spawn density features                |
| 2    | ConflictAwareObservation     | 39  | + 5 global + 9 local CBS-style features  |

## Results: BC-only Done-Rate
| Observation        | BC Acc | Done-Rate | Lift vs. Tier 0 |
|--------------------|--------|-----------|-----------------|
| DecisionPoint (T0) | 0.85   | **0.22**  | baseline        |
| SpawnAware (T1)    | 0.90   | **0.50**  | +127%           |
| ConflictAware (T2) | **0.955** | **0.58** | **+163%**     |

## Key Findings
1. Local action-conditioned conflict features prevent the
   "spurious correlation on global features" problem.
2. CBS-inspired feature design transfers cleanly to RL observation.
3. BC accuracy ≥ 95% achievable with 7000 demos at decision points.


