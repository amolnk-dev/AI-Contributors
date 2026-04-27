# Autoresearch: Feature-Driven Optimization Loop

How we used autonomous Claude agents to improve the Astar Island model from WAVG=87.36 to 88.53 (+1.17) in one overnight session.

## Overview

We ran two parallel autoresearch agents that continuously experimented with the model:

1. **Parameter Agent** — tuned existing hyperparameters (EMP_BLEND, blend weights, XGB params, L7 strengths)
2. **Feature Agent** — discovered and added new observation-derived features to XGBoost

Both agents followed the same protocol: edit train.py, run LORO backtest (~6 min), keep if improved, revert if not. The feature agent was the breakthrough — it added 39 new features that the parameter agent couldn't have found.

## The Loop

```
┌─────────────────────────────────────────────────┐
│  1. ANALYZE — Deep observation data analysis     │
│     449 observations × 9 rounds                  │
│     Correlation: obs features vs GT outcomes     │
│     → Found obs_settl_rate (r=0.41) as strongest │
│       unused predictor                           │
├─────────────────────────────────────────────────┤
│  2. HYPOTHESIZE — Generate feature candidates    │
│     Round-level: min_food, max_pop, food_std...  │
│     Per-cell: settl_rate, ruin_rate, neighbor     │
│     density, nearest settlement stats            │
├─────────────────────────────────────────────────┤
│  3. IMPLEMENT — Add feature to train.py          │
│     Enhanced obs stats: 7 → 23 features          │
│     Per-cell obs features: 0 → 23 features       │
│     Total: 37 → 76 XGBoost features              │
├─────────────────────────────────────────────────┤
│  4. BACKTEST — Run 9-fold LORO evaluation        │
│     `python train.py` → val_metric: XX.XXXX      │
│     ~6 minutes per experiment                    │
├─────────────────────────────────────────────────┤
│  5. DECIDE                                       │
│     If improved → git commit, update baseline    │
│     If worse → revert, try next hypothesis       │
├─────────────────────────────────────────────────┤
│  6. REPEAT — go to step 2                        │
└─────────────────────────────────────────────────┘
```

## What the Feature Agent Found

### Phase 1: Enhanced Round-Level Obs Stats (7 → 23 features)

The original model used 7 aggregate stats from observations:
`[avg_pop, avg_food, avg_wealth, avg_defense, alive_rate, factions/query, port_rate]`

The agent systematically added features and tested each:

| Experiment | Features Added | WAVG | Delta |
|-----------|---------------|------|-------|
| Baseline | 7 obs stats | 87.36 | — |
| +min_food, max_pop, food_std | 10 | 87.67 | +0.31 |
| +min_defense, defense_std, n_factions | 13 | 87.72 | +0.05 |
| +obs_settl_rate, obs_ruin_rate | 15 | 87.73 | +0.01 |
| +food_deficit | 16 | 87.75 | +0.02 |
| +dead_rate, wealth_std, max_food | 19 | 87.78 | +0.03 |
| +pop_std, max_food, min_pop, max_def... | 23 | 87.88 | +0.10 |

Key insight: `min_food` was the single most valuable new stat — it signals winter severity. Rounds with low min_food (R8, R10) are extinction rounds.

### Phase 2: Per-Cell Observation Features (0 → 23 features)

This was the big breakthrough. For each cell on the map, the agent computed features from the 50 viewport observations:

| Feature | Description | Why it works |
|---------|------------|-------------|
| `obs_settl_rate` | Fraction of obs runs where cell was settlement | Direct Monte Carlo estimate (r=0.41 with GT!) |
| `obs_empty_rate` | Fraction where cell was empty | Inverse signal |
| `obs_ruin_rate` | Fraction where cell was ruin | Collapse indicator |
| `obs_freq` | How often cell was observed | Coverage signal |
| `neighbor_settl_rate_r2` | Mean settlement rate in radius 2 | Spatial expansion pressure |
| `neighbor_settl_count_r2` | Count of neighbors with settl_rate > 0.1 | Expansion cluster detection |
| `max_neighbor_settl_rate_r1` | Max settl rate at distance 1 | Immediate expansion source |
| `nearest_obs_settl_pop/food/dist` | Stats of nearest observed settlement | Strength of expansion source |
| `sum_settl_r2/r3` | Sum of settlement rates in r2/r3 | Aggregated expansion signal |
| `sum_ruin_r2/r3` | Sum of ruin rates in r2/r3 | Aggregated collapse signal |
| `log_settl_count`, `log_ruin_count` | Log-scaled absolute counts | Magnitude-aware versions |
| `sum_log_settl_r2`, `sum_log_ruin_r2` | Spatial log aggregates | Neighbor magnitude |
| `dynamic_rate` | settl_rate + ruin_rate | Overall activity signal |
| `nearest_obs_settl_wealth/defense` | Economic/military stats | Trade and raid indicators |

Each feature was added one at a time, tested, and kept only if it improved LORO.

### Phase 3: Hyperparameter Co-optimization

With 76 features (vs 37 before), the optimal hyperparameters shifted:

| Parameter | Before | After | Why |
|-----------|--------|-------|-----|
| `plains` blend | 0.55 | 0.90 | More features = trust XGBoost more |
| `forest` blend | 0.65 | 0.75 | Slight increase |
| `settl` blend | 0.75 | 0.80 | Slight increase |
| `n_estimators` | 300 | 600 | More features need more trees |
| `colsample_bytree` | 0.90 | 0.55 | Feature subsampling for 76 features |

## Results

### LORO Progression (R1-R10, 9-fold cross-validation)

```
87.36  ████████████████████████████████████████████▌           Baseline
87.67  █████████████████████████████████████████████▊           +enhanced obs stats
87.92  ██████████████████████████████████████████████▊          +per-cell obs features
88.05  ███████████████████████████████████████████████▍         +log features + spatial
88.31  ████████████████████████████████████████████████▍        +plains=0.90
88.36  ████████████████████████████████████████████████▋        +more obs stats
88.46  █████████████████████████████████████████████████        +colsample=0.65
88.53  █████████████████████████████████████████████████▎       +colsample=0.55 (BEST)
```

### Per-Round Improvements

| Round | Type | Before | After | Delta |
|-------|------|--------|-------|-------|
| R1 | Medium expansion | 86.48 | 87.08 | +0.60 |
| R2 | High expansion | 89.15 | 90.01 | +0.86 |
| R4 | Extinction | 93.44 | 94.02 | +0.58 |
| R5 | Medium | 84.93 | 85.82 | +0.89 |
| R6 | Very high expansion | 85.47 | 86.65 | +1.18 |
| R7 | Extreme expansion | 70.70 | 71.57 | +0.87 |
| R8 | Extinction | 94.03 | 94.72 | +0.69 |
| R9 | Medium | 91.85 | 92.42 | +0.57 |
| R10 | Extinction | 89.71 | 93.42 | +3.71 |

R10 (extinction) and R6 (expansion) benefited most — the per-cell obs features capture round-specific dynamics that the heuristic tables miss.

### Production Backtest (R11)

When we ported the agent's improvements to production and backtested against R11:

| Config | R11 avg | R11 weighted |
|--------|---------|-------------|
| Submitted (old model) | 85.09 | 138.6 |
| Agent model (76 features) | 87.62 | 142.7 |
| Improvement | **+2.53** | **+4.1** |

## How to Run

### Launch the autoresearch agent

```bash
# The agent edits train.py and runs LORO backtests autonomously
# Launch from Claude Code:
# Agent prompt should include:
#   - Current baseline WAVG
#   - Path to train.py and run command
#   - List of what to try (features, params)
#   - Protocol: edit → run → commit if improved → revert if not
```

### The agent's eval command

```bash
cd tasks/astar-island && python train.py
# Output: val_metric: XX.XXXXXX (higher = better)
# Time: ~6 minutes per experiment (9-fold LORO)
```

### Port improvements to production

After the agent finds improvements in train.py, port to production:

1. Copy `enhanced_obs_stats()` → model.py's `compute_obs_stats()`
2. Copy `compute_cell_obs_features()` → model.py
3. Update `gbt_predict()` in model.py to accept new features
4. Update blend weights in `build_prediction()`
5. Update retrain_gbt.py to match
6. Run `python retrain_gbt.py` to rebuild models
7. Verify: `python run.py --dry-run`

## Key Learnings

1. **Feature engineering > parameter tuning.** The first 39 new features gave +0.85 WAVG. The subsequent 20 parameter experiments gave +0.32.

2. **Per-cell observation features are the biggest lever.** `obs_settl_rate` (r=0.41 with GT) is a direct Monte Carlo estimate — exactly what the competition measures. We had this data all along but weren't using it.

3. **Correlation analysis guides feature selection.** Running `analyze_observations.py` first identified which features would actually help before spending compute testing them.

4. **Blend weights must co-evolve with features.** Adding features made XGBoost stronger, so optimal blend weight for plains jumped from 0.55 to 0.90. Testing features without re-tuning blend would underestimate their value.

5. **Two agents can conflict.** Running two agents editing the same file caused race conditions (one sets colsample=0.45, other restores 0.55). Better to run one agent at a time, or use worktrees for true isolation.

6. **colsample_bytree scales inversely with feature count.** With 76 features, 0.55 colsample (using ~42 features per tree) outperforms 0.90 (using ~68). More features need more subsampling to prevent overfitting.

## Files

| File | Role |
|------|------|
| `train.py` | Autoresearch target — agent edits this |
| `model.py` | Production pipeline — port improvements here |
| `retrain_gbt.py` | Retrain XGBoost with new features |
| `analyze_observations.py` | Deep analysis → feature candidates |
| `results.tsv` | Experiment log (all attempts) |
| `program.md` | Agent instructions |
