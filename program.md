# Autoresearch Program — Astar Island

## Task Selection

This task optimizes a probabilistic prediction model for a Norse civilization simulator.
Each eval takes ~500 seconds (13-fold LORO with XGBoost retraining). No GPU required.

## Setup

1. **Branch**: `git checkout -b autoresearch/<tag>` from `autoresearch/r16`
2. **Read**: `train.py` (eval harness + tunable params), `model.py` (production pipeline, READ ONLY)
3. **Baseline**: `cd tasks/astar-island && python train.py > run.log 2>&1`
4. **Init results.tsv**: header + baseline entry

## In-scope files

- `train.py` — **MUTABLE**. The ONLY file you edit. Contains tunable parameters at the top and the LORO evaluation loop. Prints `val_metric:` (weighted avg, higher is better, max 100).
- `model.py` — **READ ONLY**. Production prediction engine. Read to understand the pipeline, but do NOT edit.
- `dtos.py` — **READ ONLY**. Terrain constants (PROB_FLOOR, TERRAIN_TO_CLASS, etc.)
- `data/` — **READ ONLY**. Ground truth, models, observations.

## Current model architecture (in train.py)

The LORO evaluation builds predictions in layers for each held-out round:

1. **Layer 1 (Static)**: Distance-based coastal/inland priors from GT tables (calibrated R1-R15, 65 maps)
2. **Layer 3 (Context)**: adj_forests, coastal, settlement adjacency priors for all dynamic cells
3. **Layer 6 (GBT Blend)**: Terrain-specific XGBoost (76 features: 30 cell + 23 obs stats + 23 per-cell obs)
   - Per-terrain blend weights: plains=1.00, forest=1.00, settl=1.00
   - SKIPPED for rounds with 0 observations (R12 excluded from eval)
4. **Layer 6.7 (Cross-seed Empirical)**: Pool observations across all seeds, group by (terrain, distance, coastal), blend at EMP_BLEND=0.50
5. **Layer 7 (L7 Correction)**: Observation ratio correction — adjusts predictions to match observed terrain frequencies
   - Strengths: [1.40, 1.00, 0.0, 0.0, 1.50, 0.0] (zero for port/ruin)
   - Per-class clamp: empty [0.80-1.25], settl [0.40-1.30], forest [0.80-1.25]
   - Obs count gating: [200, 50, 30, 20, 100, 0]
6. **Layer 8 (Per-cell Empirical)**: Blend model with per-cell observation frequencies (MIN_SAMPLES=50, effectively disabled)

### Feature alignment (CRITICAL)

Train and inference use IDENTICAL extractors imported from model.py:
- `compute_obs_stats()` → 23 features
- `compute_cell_obs_features()` → 23 features per cell
- `_extract_cell_features()` → 30 features per cell
- **Total: 76 features**

## Tunable parameters in train.py

Located in the `TUNABLE PARAMETERS` section at the top of train.py:

```python
BLEND_WEIGHT = 0.35                    # Global GBT blend (overridden by per-terrain)
BLEND_TERRAIN = {"plains": 1.00, "forest": 1.00, "settl": 1.00}
L7_STRENGTHS = np.array([1.40, 1.00, 0.0, 0.0, 1.50, 0.0])
L7_MIN_OBS = np.array([200, 50, 30, 20, 100, 0])
L7_ADJ_MIN = 0.80
L7_ADJ_MAX = 1.25
XGB_HPARAMS = {per-terrain dicts with n_estimators=600, max_depth=5, learning_rate=0.08,
               reg_alpha=0.1, reg_lambda=2.0, subsample=0.9, colsample_bytree=0.55, min_child_weight=3}
```

Also tunable within evaluate_loro():
- `EMP_BLEND = 0.50` — weight for cross-seed empirical tables
- L7 per-class clamp bounds (adj_min, adj_max arrays)

## Current baseline: WAVG=89.39 (2026-03-21)

### Per-round LORO scores (weakest first)
```
R7  = 72.54  ← extreme expansion, sharp distance cutoff (BIGGEST OPPORTUNITY)
R1  = 85.55  ← medium expansion
R14 = 85.67  ← medium
R5  = 86.52  ← medium expansion
R6  = 87.56  ← very high expansion
R11 = 87.74  ← very high expansion
R2  = 92.14  ← high expansion
R15 = 92.42  ← medium
R10 = 92.99  ← total extinction
R9  = 93.08  ← medium
R13 = 93.89  ← medium-low
R4  = 94.26  ← extinction
R8  = 95.26  ← extinction
```

**Pattern: Strong on extinction rounds, weak on expansion rounds.**

## What to try

### High potential (+0.50 to +3.00)
- **Round-type detection**: Classify expansion vs extinction from obs_stats, use different parameters
- **Adaptive distance tables**: Scale tables based on detected expansion rate
- **Per-round-type blend weights**: Trust XGBoost more on expansion rounds
- **Expansion pressure gradient**: Non-linear distance decay for high-expansion rounds

### Medium potential (+0.10 to +0.50)
- Fine-tune EMP_BLEND (try 0.40, 0.45, 0.48, 0.52, 0.55)
- Fine-tune per-terrain blend weights (sweep ±0.05 on each)
- L7 strength tuning per class
- colsample_bytree: 0.45, 0.50, 0.55, 0.60
- Per-terrain XGB hyperparameters (different depth/lr per terrain)

### New feature ideas (+0.10 to +1.00)
- Terrain diversity in radius 3/5 (count distinct terrain types)
- Settlement viability: food access / competition ratio
- Port BFS distance over passable terrain
- Faction diversity per cell radius (conflict zone)
- Defense variance across observations (raiding indicator)
- Wealth depletion rate (raiding/no-trade signal)
- Forest reclamation rate from obs grids

### Low potential (+0.00 to +0.10)
- XGB learning rate (0.05, 0.10)
- n_estimators (700, 800 — slower)
- L7 MIN_OBS thresholds
- SMOOTH_WEIGHT: 0.01-0.03 (currently disabled)

## What NOT to do (proven failures)

- KNN round matching (-9.80)
- Probability sharpening (-0.28)
- LightGBM (-0.33)
- Spatial L7 BFS (-0.39)
- Viability scoring (-0.15 to -0.66)
- Cross-seed transfer Layer 4 (-0.50)
- Layer 2 Bayesian observations (adds noise)
- Increase PROB_FLOOR (catastrophic)
- Simulator blend (+0.00, 30s overhead)
- n_ports feature (no improvement)
- Voronoi territory size (non-monotonic, no signal)
- Center of mass distance (near-zero correlation)
- Residual learning (worse than direct prediction)
- Global features like settl_ratio/forest_ratio (R7=68 but R2/R6 drop)

## Run command

```bash
cd tasks/astar-island && python train.py > run.log 2>&1
grep "^val_metric:" run.log
```

Time budget: ~500 seconds per eval (13-fold LORO with XGBoost retraining per fold).

## The experiment loop

LOOP FOREVER:

1. Look at git state and results.tsv for context
2. Edit `train.py` tunable parameters section
3. `git add tasks/astar-island/train.py && git commit -m "experiment: <description>"`
4. `cd tasks/astar-island && python train.py > run.log 2>&1`
5. `grep "^val_metric:" run.log`
6. If empty/crash: `tail -n 50 run.log` to debug
7. Log to results.tsv: `commit\tval_metric\tmemory_gb\tdiff_lines\tstatus\tdescription\treject_reason`
8. If improved: `git add tasks/astar-island/results.tsv && git commit --amend --no-edit`
9. If worse: record hash+result in results.tsv, then `git checkout -- tasks/astar-island/train.py` (revert)

## Scoring

score = 100 * exp(-3 * entropy_weighted_KL_divergence)
- 100 = perfect, 0 = terrible
- Only dynamic cells (non-zero entropy in ground truth) count, weighted by entropy
- NEVER let any probability be 0.0 (KL divergence -> infinity)
- Entropy weighting means high-uncertainty cells matter most

## Data available

13 rounds with GT + observations: R1, R2, R4-R11, R13-R15
R12: GT only (0 observations) — excluded from LORO
R3: excluded (different dynamics)
