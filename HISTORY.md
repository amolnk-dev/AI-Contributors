# Astar Island — Competition History & Learnings

## Score Trajectory

| Round | Live Score | Rank | Teams | Weight | Weighted | Key Changes |
|-------|-----------|------|-------|--------|----------|-------------|
| R1 | 8.17 | 89/117 | 117 | 1.00 | 8.17 | First attempt, many bugs |
| R2 | 81.37 | 23/153 | 153 | 1.05 | 85.44 | Fixed submission pipeline |
| R3 | — | — | — | — | — | Didn't participate |
| R4 | 91.80 | **5**/86 | 86 | 1.16 | 106.41 | Coastal/inland split (+9 pts!) |
| R5 | 80.36 | 18/144 | 144 | 1.22 | 97.72 | New map type, model struggled |
| R6 | 84.91 | 8/186 | 186 | 1.28 | 108.52 | GBT ensemble added |
| R7 | 63.19 | 57/199 | 199 | 1.34 | 84.68 | Extreme expansion — L7 destroyed rare classes |
| R8 | 76.12 | 75/214 | 214 | 1.41 | 107.16 | Extinction round, model over-predicted settlements |
| R9 | 91.77 | 17/221 | 221 | 1.48 | 135.66 | Cross-seed empirical tables deployed |
| R10 | 75.72 | 87/238 | 238 | 1.55 | 117.49 | Total extinction — model retrained with obs features |
| R11 | 85.09 | 35/171 | 171 | 1.63 | 138.60 | Good, but model hadn't been retrained with R8-R10 |
| R12 | — | — | — | — | — | Didn't participate (0 observations) |
| R13 | **93.94** | **1**/186 | 186 | 1.79 | **168.44** | Peak rank! Full 76-feature model + autoresearch tuning |
| R14 | 85.85 | 8/244 | 244 | 1.88 | 161.69 | Solid |
| R15 | 92.31 | 28/262 | 262 | 1.98 | 182.62 | Feature alignment bug still present |
| R16 | 88.85 | **4**/272 | 272 | 2.08 | 184.69 | Feature alignment FIX deployed |
| R17 | **92.99** | 21/283 | 283 | 2.18 | **202.92** | **Best weighted score!** Terrain diversity feature |

**Leaderboard score = MAX(weighted) = 202.92 (R17)**

## What Worked (keep doing)

### 1. Coastal/inland distance split (+9 pts, R4)
The single biggest improvement. Cells with 2+ ocean neighbors behave differently (ports possible). Separate distance tables for coastal vs inland doubled the model's accuracy.

### 2. XGBoost per-terrain specialists (+3 pts cumulative)
Training separate models for plains, forest, and settlement cells. Each terrain type has different dynamics — a universal model blurs them.

### 3. Per-cell observation features (+0.85 WAVG)
Using the 50 viewport observations as direct Monte Carlo estimates. `obs_settl_rate` (r=0.41 with GT) was the single most predictive feature. 23 per-cell features total.

### 4. Feature alignment fix (+3.6 WAVG in LORO)
Train and inference were using different feature extractors (77 vs 76 features). Fixing this so both use identical functions from model.py was a critical bug fix.

### 5. Safe L7 — zero correction for rare classes
Port (cls=2) and ruin (cls=3) should NEVER be corrected by L7. Observation under-sampling + KL asymmetry (60x more punishing for under-prediction) means even small corrections on rare classes can destroy the score.

### 6. Terrain diversity r3 (+0.06 WAVG)
Count of distinct terrain types in radius 3. Cells at biome boundaries have different expansion dynamics. Small but reliable improvement.

### 7. Cross-seed empirical distance tables (+1 pt)
Pool ALL observations across ALL seeds, group by (terrain, distance, coastal). Adapts to each round's hidden parameters at the distance level.

### 8. Excluding R12 from training (+3.6 WAVG)
R12 had 0 observations. Training XGBoost on it with all-zero observation features polluted the model. Excluding it cleaned up the training data.

### 9. Vectorization of feature extraction (500s → 360s LORO)
Replaced nested Python loops with scipy.ndimage.convolve and distance_transform_cdt. Made autoresearch 28% faster.

### 10. Batch XGBoost prediction (per-cell → per-terrain batch)
Instead of 7200 individual predict calls, group cells by terrain type and batch predict. 125s → 3s per fold.

## What Failed (don't try again)

| Experiment | Delta | Why it failed |
|-----------|-------|---------------|
| KNN round matching | -9.80 | Catastrophic overfitting to nearest round |
| Probability sharpening | -0.28 | Increased KL on uncertain cells |
| LightGBM | -0.33 | Worse than XGBoost on this data size |
| Spatial L7 BFS | -0.39 | Over-complicated, no signal |
| Viability scoring | -0.66 | Wrong proxy for settlement survival |
| Cross-seed transfer (L4) | -0.50 | Cross-seed observations are too noisy |
| Layer 2 Bayesian obs | noise | With ~10 obs per cell, raw frequencies are noise |
| Higher PROB_FLOOR | catastrophic | KL divergence extremely sensitive to floor |
| Simulator blend | +0.00 | 30s overhead, zero improvement — too noisy |
| n_ports feature | +0.00 | No signal |
| Voronoi territory | +0.00 | Non-monotonic, no correlation |
| Center of mass distance | +0.00 | Near-zero correlation with GT |
| Residual learning | -0.40 | Worse than direct prediction |
| Grid-level features | +0.00 | Constant across cells = useless for per-cell XGB |
| Wider L7 clamp | -0.17 | Hurts R10/R11 more than helps R7 |
| Plains in r3 | -0.39 | Too correlated with existing features |
| Settlement viability | -0.06 | forests_r2/(1+settl_r3) — no marginal signal |
| Feature caching in LORO | OOM crash | Holding all 14 rounds in memory killed Python |

## Score Patterns

### Strong on extinction rounds
R4=91.8, R9=91.8, R13=93.9, R17=93.0. The model's distance tables + L7 correction handle these well.

### Weak on expansion rounds
R7=63.2 (live), R5=80.4. High expansion rounds have settlements spreading far beyond what distance tables predict. R7 LORO is 72.5 — still the weakest.

### Best performance on later rounds
R13 (rank 1), R16 (rank 4), R17 (rank 21) — model improves as we accumulate more training data and fix bugs. Round weights favor later rounds (1.05^n).

## Current Model (as of R17)

```
77 features: 31 cell + 23 obs stats + 23 per-cell obs

Prediction Pipeline:
  L1: Static distance tables (coastal/inland × 4 terrain types)
  L3: Context priors (adj_forests, coastal, settlement adjacency)
  L6: XGBoost blend (per-terrain: plains=1.0, forest=1.0, settl=1.0)
  L6.7: Cross-seed empirical distance tables (EMP_BLEND=0.50)
  L7: Safe observation ratio correction (zero for port/ruin)
  L8: Per-cell empirical (disabled, MIN_SAMPLES=50)

XGBoost: 600 trees, depth 5, lr 0.08, colsample 0.55
Trained on: 15 rounds (R1-R17 excl R3,R12), 102K samples
LORO baseline: WAVG ~89.5 (15-fold)
```

## What to Improve Next

### 1. R7 (LORO=72.5) — biggest single opportunity
R7 is extreme expansion. Settlements spread much further than average. Ideas:
- **Round-type detection**: classify expansion vs extinction from obs_stats
- **Per-type parameters**: different blend/L7/EMP for expansion vs extinction
- **Non-linear distance decay**: exponential instead of linear buckets

### 2. CNN/U-Net for spatial patterns
XGBoost treats cells independently. Can't learn: "settlements form corridors between ports" or "expansion follows coastline". A spatial model could add +1-3 WAVG.

### 3. Better query allocation
50 queries across 5 seeds = ~10 per seed. Concentrating on high-entropy areas or repeating viewports on the same area could enable Layer 8 (per-cell empirical, currently disabled).

### 4. Port corridor features
Cells between two ports may have different dynamics (trade routes, coastal expansion). Feature: min(BFS to port A + BFS to port B).

### 5. Faction analysis
Observations include `owner_id`. Tracking faction diversity, conquest events, and territory boundaries could signal conflict zones.

### 6. Autoresearch swarm findings
3 GCP VMs running Gemini-guided experiments. Check results:
```bash
bash scripts/gcp/collect-astar.sh
bash scripts/gcp/collect-astar.sh --best  # download best train.py
```

## Infrastructure

| Resource | Purpose |
|----------|---------|
| Local Mac (M1 Pro) | Main development, feature autoresearch (~360s/LORO) |
| ainm-astar-autoresearch (176 CPUs) | Swarm VM, parallel XGB (~189s/LORO) |
| ainm-astar-swarm-a (44 CPUs) | Swarm worker |
| ainm-astar-swarm-b (44 CPUs) | Swarm worker |
| `collect-astar.sh` | Orchestrator: pull/merge/show swarm results |
| `autoresearch_swarm.py` | Gemini-guided autonomous experiment loop |
