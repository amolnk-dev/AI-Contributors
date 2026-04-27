# Astar Island Round Pipeline

Step-by-step for each new round. ~5 min per round.

## When a round opens

### 1. Submit (2 min)
```bash
cd tasks/astar-island
ASTAR_TOKEN=<token> python run.py
```
This queries 50 observations, builds predictions with the current model, and submits all 5 seeds.
Uses ALL 50 observations across all seeds for L7 correction and empirical tables.

### 2. Check observation stats (30 sec)
```bash
ASTAR_TOKEN=<token> python3 -c "
from client import AstarClient
c = AstarClient()
rounds = c.get_rounds()
for r in rounds:
    if r.status == 'active' or r.round_number >= 9:
        print(f'R{r.round_number}: {r.status}')
"
```

### 3. Gauge the round (30 sec)
After submission, check `settl_per_obs` in the logs. Compare to training data:

| Round | settl/obs | Expansion | Type |
|-------|-----------|-----------|------|
| R1 | 38.6 | Medium | Archipelago |
| R2 | 43.9 | High | Archipelago |
| R4 | 22.1 | Low | Continental |
| R5 | 29.3 | Medium | Continental |
| R6 | 57.1 | Very high | Continental |
| R7 | 35.9 | Medium | Continental (sharp cutoff) |
| R8 | 7.5 | Very low | Extreme extinction |
| R9 | ~30 | Medium | Continental |
| R10 | 2.8 | Very low | Extreme extinction |

If settl/obs is within 20-60 range, the model is in-distribution. Outside that = novel round.
R8 and R10 are extinction rounds: all settlements died, GT is ~75% empty + 25% forest.

## When a round completes (scoring done)

### 4. Download GT + save initial states (1 min)
```bash
ASTAR_TOKEN=<token> python3 -c "
from client import AstarClient
import numpy as np, json
c = AstarClient()
ROUND_NUM = 16  # <-- change this
rounds = c.get_rounds()
round_id = [r for r in rounds if r.round_number==ROUND_NUM][0].id
detail = c.get_round_detail(round_id)

# Save initial states
states = {'initial_states': []}
for seed in range(5):
    s = detail.initial_states[seed]
    states['initial_states'].append({'grid': s.grid, 'settlements': [x.model_dump() for x in s.settlements]})
with open(f'data/round{ROUND_NUM}_initial.json', 'w') as f:
    json.dump(states, f)

# Save GT
for seed in range(5):
    a = c.get_analysis(round_id, seed)
    np.save(f'data/gt_r{ROUND_NUM}_seed{seed}.npy', np.array(a.ground_truth))
    print(f'Seed {seed}: score={a.score:.2f}')
print(f'Round {ROUND_NUM} GT saved.')
"
```

### 5. Update ROUNDS + retrain GBT (2 min)
1. Add round ID to `ROUNDS` dict in both `train.py` and `retrain_gbt.py`
2. Add round weight to `ROUND_WEIGHTS` in both files
3. Update `test_rounds` list in `evaluate_loro()` in `train.py`
4. **Only add rounds with observations** — rounds with 0 obs (like R12) should be excluded
5. Retrain: `python retrain_gbt.py`

### 6. Run LORO to verify (~8 min with 13 rounds)
```bash
python train.py
```
Check that val_metric improved or stayed stable with the new data.

## Model Architecture (current, WAVG=89.39 baseline, 2026-03-21)

```
PREDICTION PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  API Query (50 budget)
       │
       ▼
  ┌─────────────┐     ┌──────────────────┐
  │  Viewport   │────▶│  Observations    │
  │  15×15 grid │     │  (JSONL files)   │
  └─────────────┘     └────────┬─────────┘
                               │
    ┌──────────────────────────┘
    ▼
  build_prediction()
  │
  ├─ L1: build_static_prediction()
  │   └── Distance tables calibrated from R1-R15 GT (65 maps)
  │       └── Coastal/inland split, distance buckets 1-8+99
  │
  ├─ L3: fill_unobserved_dynamic()
  │   └── Context priors (adj_forests, coastal, settlement adjacency)
  │
  ├─ L6: gbt_predict() — XGBoost blend
  │   └── 76 features: 30 cell + 23 obs stats + 23 per-cell obs
  │   └── Per-terrain blend: plains=0.90, forest=0.75, settl=0.80
  │   └── Terrain-specific models (plains/forest/settlement)
  │   └── 3 terrain × 6 class regressors = 18 XGBoost models
  │   └── Hparams: 600 trees, depth 5, lr 0.08, subsample=0.9
  │   └── SKIPPED for rounds with 0 observations
  │
  ├─ L6.7: Cross-seed empirical distance tables
  │   └── Pool ALL observations across ALL seeds
  │   └── Group by (initial_terrain, distance_bucket, coastal)
  │   └── EMP_BLEND = 0.40
  │
  ├─ L7: Safe L7 observation ratio correction
  │   └── Strengths: [1.20, 0.80, 0.0, 0.0, 1.30, 0.0]
  │   └── Per-class clamp: empty [0.80-1.25], settl [0.40-1.30]
  │   └── Obs count gating: [200, 50, 30, 20, 100, 0]
  │   └── Zero for port (cls=2) and ruin (cls=3)
  │
  ├─ L8: Per-cell empirical (MIN_SAMPLES=50, effectively disabled)
  │
  └── normalize_prediction() → Submit
```

### Feature alignment (CRITICAL)

Train (`train.py`) and inference (`model.py`) MUST use identical feature extractors:
- `compute_obs_stats()` from model.py → 23 features
- `compute_cell_obs_features()` from model.py → 23 features per cell
- `_extract_cell_features()` from model.py → 30 features per cell
- **Total: 76 features per cell**

`train.py` imports these functions from `model.py` — never define duplicate extractors.

### LORO evaluation

- **Rounds**: R1, R2, R4-R11, R13-R15 (13 rounds, R12 excluded: 0 obs)
- **Method**: Leave-One-Round-Out — for each held-out round, train on other 12
- **Time**: ~500s (~8 min) on Apple Silicon
- **Metric**: `val_metric` = weighted average (1.05^(round-1))
- **Baseline**: WAVG=89.39, unweighted=89.20 (2026-03-21)

### Per-round LORO scores (baseline 2026-03-21)

| Round | Score | Notes |
|-------|-------|-------|
| R1 | 85.55 | |
| R2 | 92.14 | |
| R4 | 94.26 | |
| R5 | 86.52 | |
| R6 | 87.56 | |
| R7 | 72.54 | Weakest — extinction round |
| R8 | 95.26 | Best — similar to R16 profile |
| R9 | 93.08 | |
| R10 | 92.99 | |
| R11 | 87.74 | |
| R13 | 93.89 | |
| R14 | 85.67 | |
| R15 | 92.42 | |

### Autoresearch

```bash
# Run LORO (prints val_metric for autoresearch agent)
cd tasks/astar-island && python train.py

# Tunable parameters in train.py:
# - BLEND_WEIGHT, BLEND_TERRAIN — XGBoost blend weights
# - L7_STRENGTHS, L7_MIN_OBS, L7_ADJ_MIN, L7_ADJ_MAX — L7 correction
# - XGB_HPARAMS — per-terrain XGBoost hyperparameters
# - EMP_BLEND (in evaluate_loro) — empirical table blend weight

# Retrain production model after improvements:
python retrain_gbt.py
```

## Key Files

| File | Purpose |
|------|---------|
| `run.py` | Main entry: query + predict + submit |
| `model.py` | Prediction pipeline (production, ~1400 lines) |
| `train.py` | LORO evaluation (autoresearch-compatible) |
| `retrain_gbt.py` | Retrain XGBoost on all rounds |
| `evaluate.py` | Scoring function |
| `simulator.py` | Monte Carlo simulator (disabled, SIM_BLEND=0.0) |
| `data/gbt_models.pkl` | Trained XGBoost models (76 features) |
| `data/obs_<round_id>.jsonl` | Saved observations per round |
| `data/gt_r<N>_seed<S>.npy` | Ground truth per round/seed |
| `data/round<N>_initial.json` | Initial states per round |
| `results.tsv` | Experiment log |

## Key Learnings

1. **Leaderboard = MAX(round_score x round_weight)** — only best weighted round matters
2. **Can resubmit** — only last submission counts
3. **Use ALL 50 observations** for L7/empirical tables — same-seed only cost 13 pts on R10
4. **Settlement stats (population, food, wealth, defense)** are 40-63% of XGBoost feature importance
5. **Safe L7** — never correct port/ruin classes (observation under-sampling + KL asymmetry)
6. **Feature alignment** — train.py MUST use same extractors as model.py (76 features)
7. **Exclude no-obs rounds** — R12 has 0 observations, training on it pollutes XGBoost
8. **PROB_FLOOR=0.0001** — higher floors hurt badly
9. Each round has 50 queries, 5 seeds, 15x15 viewport, 40x40 map
10. Round weights: 1.05^(round-1) — later rounds worth more
11. R8 and R10 are extinction rounds — model handles these via L7 correction
12. LORO takes ~8 min with 13 rounds on Apple Silicon
