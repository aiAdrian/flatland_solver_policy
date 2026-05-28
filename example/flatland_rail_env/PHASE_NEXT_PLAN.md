# Roadmap: 100% done + Cost-Optimierung + Scaling

## Ziel
Phase 1: done = 100% (matches DLA)
Phase 2: cost/episode_len < DLA (besser als baseline)
Phase 3: Generalisierung 1, 2, 5, 10, 15, 20 Agenten

## Vorhandene Bausteine ✅
- DeadLockAvoidancePolicy (DLA) — instanziert in train_marl.py:310
- il_collect_demos.py — Demo-Collector mit DLA-Teacher
- BC-Pretraining — funktioniert (bc_checkpoint_*.pt)
- MAPPO + ConflictAware Obs — funktioniert (0.75 done erreicht)
- PPO-Bugfix (KL-clamp + early-stop) — heute Nacht eingebaut

## Schritt-für-Schritt Plan (~1 Woche Arbeit)

### Step 1: DLA-Action als Observation-Feature (1 Tag)
File: conflict_aware_observation.py
Add: 5 features für DLA-Action one-hot
Test: Quick BC + 100 Eps MAPPO → erwartet done > 0.85

### Step 2: BC mit DLA-Imitation (statt random) (0.5 Tag)
File: il_collect_demos.py + bc_train.py
Mod: BC lernt aus DLA-Trajectories statt aus Random
Test: BC-only-Eval → erwartet done = 0.85-0.95

### Step 3: MAPPO mit BC-warmstart + DLA-feature (0.5 Tag)
Train: 200 Eps mit allen Features
Erwartet: done = 0.90-0.98

### Step 4: Reward-Shaping für Cost-Optimierung (1 Tag)
File: rewards.py
Mod: belohne early arrival, penalty unnötiges STOP
Test: 100 Eps → erwartet episode_len -10-20%

### Step 5: Inter-Agent Attention (optional, 1-2 Tage)
File: mappo_policy.py — neue AttentionLayer
Test: nur falls Step 1-4 nicht 100% done erreicht

### Step 6: Scaling Eval (0.5 Tag)
For N in [1, 2, 5, 10, 15, 20]:
    eval --n_agents N --episodes 50

## Erwartete Resultate
- Phase 1: done = 95-100% bei 5 Agenten
- Phase 2: cost = 0.85x DLA bei gleicher done_rate
- Phase 3: degrades ab 15+ Agenten (architecture limit)

## Risiko-Backup
Champion bleibt: mappo_FINAL_done75_20260528.pt (done=0.75)
Nichts wird überschrieben — alle neuen runs in subdirs.
