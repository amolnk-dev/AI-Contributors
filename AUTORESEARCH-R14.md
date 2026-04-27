# Autoresearch Session: R14 Era (2026-03-21)

## What We're Doing

Running 3 parallel autonomous Claude agents, each in an isolated git worktree,
to improve the Astar Island prediction model. Each agent follows the same
autoresearch protocol but targets a different improvement axis.

## The Autoresearch Pattern

Every agent follows the exact same loop:

```
┌──────────────────────────────────────────────────────────────┐
│                    AUTORESEARCH LOOP                          │
│                                                              │
│  1. HYPOTHESIZE — pick ONE thing to try                      │
│     (a feature, a parameter, a table change)                 │
│                                                              │
│  2. IMPLEMENT — edit train.py                                │
│     (train.py is the ONLY mutable file)                      │
│                                                              │
│  3. BACKTEST — run: python3 train.py                         │
│     Output: val_metric: XX.XXXXXX (weighted avg, higher=better) │
│     Time: ~8 minutes (12-fold LORO cross-validation)         │
│                                                              │
│  4. DECIDE                                                   │
│     If val_metric > baseline:                                │
│       → git commit with message:                             │
│         "experiment: <description> (+X.XXX WAVG=YY.YYY)"    │
│       → update baseline                                     │
│     If val_metric <= baseline:                               │
│       → git checkout train.py (revert)                       │
│       → try next hypothesis                                  │
│                                                              │
│  5. REPEAT — go to step 1                                    │
│                                                              │
│  RULES:                                                      │
│  - Only edit train.py (never model.py or run.py)             │
│  - One change at a time (isolate what helped)                │
│  - Always record the delta and new WAVG in commit message    │
│  - If 3 consecutive experiments fail, try a different axis   │
│  - Never revert a successful experiment                      │
└──────────────────────────────────────────────────────────────┘
```

## Current Baseline

- **WAVG: 88.83** (12-fold LORO, R1-R13 excluding R12)
- **Features: 90** in train.py (30 cell + 39 enhanced obs stats + 21 cell obs)
- **XGB: n_est=600, depth=5, lr=0.08, colsample=0.55**
- **Blend: plains=0.90, forest=0.75, settl=0.80**

### Per-Round Scores (weakest first)
```
R7  = 71.80  ← extreme expansion, sharp distance cutoff
R1  = 86.00  ← medium expansion
R11 = 86.12  ← very high expansion (25.8% settl rate)
R6  = 86.53  ← very high expansion (24.4% settl rate)
R5  = 86.65  ← medium expansion
R2  = 92.03  ← high expansion
R10 = 91.69  ← total extinction
R9  = 93.09  ← medium
R13 = 93.35  ← medium-low
R8  = 93.94  ← extinction
R4  = 94.20  ← extinction
```

**Pattern: We're strong on extinction rounds, weak on high expansion rounds.**

## The Three Agents

### Agent 1: Expansion Fixer

**Goal:** Improve scores on R6 (86.5), R7 (71.8), R11 (86.1) — our 3 weakest rounds.

**Why these are weak:** High expansion rounds have settlements spreading far from
initial positions. Our distance-based heuristic tables were calibrated on all rounds
including extinction rounds, which biases toward less expansion.

**Ideas to try:**
- Recalibrate static distance tables using only expansion rounds (R2, R6, R7, R11)
  vs extinction rounds (R4, R8, R10) — adaptive tables based on detected round type
- Coastal expansion patterns: ports cluster near other ports, model port-to-port spread
- Settlement density-adaptive blend weights: trust XGBoost more on expansion rounds
- Per-round-type distance tables (detect expansion vs extinction from obs stats,
  switch between two sets of tables)
- Expansion pressure gradient: model how settlement probability decays differently
  when many settlements are nearby vs few

**Observable signals for expansion detection:**
- obs_settl_rate > 0.15 → expansion round
- port_count > 80 → high expansion
- faction_count > 50 → expansion
- avg_population > 1.1 → growth-favored

### Agent 2: Hidden Parameter Detector

**Goal:** Extract more information from observations about the simulation's hidden parameters.

**The simulation has hidden forces we're not fully modelling:**

#### Raiding Mechanics
- Settlements raid each other; longships extend range
- Desperate settlements (low food) raid more aggressively
- Successful raids loot resources and damage defender
- Conquered settlements can change faction (owner_id)

**Feature ideas:**
- Defense variance across observations (high variance = active raiding)
- Wealth depletion rate (wealth near 0 = heavy raiding or no trade)
- Faction diversity per cell radius (many factions nearby = conflict zone)
- Owner_id change frequency (from obs — same settlement, different owner = conquest)

#### Trade Mechanics
- Ports within range trade if not at war
- Trade generates wealth and food
- Technology diffuses between trading partners

**Feature ideas:**
- Port-to-port distance features (nearest port pair, port cluster density)
- Wealth concentration near ports (trade indicator)
- Tech level proxy (wealth × defense correlation)

#### Environmental Mechanics
- Ruins get reclaimed by forest or nearby settlements
- Coastal ruins can become ports

**Feature ideas:**
- Forest reclamation rate from obs grids (count cells that were ruin → forest)
- Ruin lifetime proxy (ruin persistence across observations)
- Coastal ruin → port conversion rate

#### Winter Severity
- Each year has varying severity
- Settlements collapse from starvation

**Feature ideas:**
- min_food already captured, but food_variance could indicate winter unpredictability
- Population crash rate (delta between obs population snapshots)

### Agent 3: Parameter Tuner

**Goal:** Fine-tune hyperparameters, especially after agents 1 and 2 add new features.

**Ideas to try:**
- colsample_bytree sweep: 0.45, 0.50, 0.55, 0.60 (currently 0.55)
- min_child_weight: 2, 3, 4, 5 (currently 3)
- learning_rate: 0.06, 0.08, 0.10 (currently 0.08)
- max_depth: 4, 5, 6 (currently 5)
- n_estimators: 500, 600, 700, 800 (currently 600)
- reg_alpha: 0.05, 0.1, 0.2 (currently 0.1)
- reg_lambda: 1.5, 2.0, 2.5 (currently 2.0)
- Per-terrain blend weights: sweep plains 0.85-0.95, forest 0.70-0.80, settl 0.75-0.85
- EMP_BLEND: 0.45, 0.50, 0.55 (currently 0.50)
- L7_STRENGTHS: adjust per-class strengths and clamps
- SMOOTH_WEIGHT: 0.01, 0.02, 0.03 (currently 0.0 — might help on extreme rounds)

## How Worktrees Work

Each agent runs in an isolated copy of the repo:

```
main (baseline)
  ├── worktree-1/ → Agent 1 (expansion fixer)
  ├── worktree-2/ → Agent 2 (hidden params)
  └── worktree-3/ → Agent 3 (parameter tuner)
```

No conflicts — each agent's changes are on its own branch.

## After the Agents Finish

1. **Compare:** Look at each agent's final WAVG and per-round breakdown
2. **Cherry-pick:** Take the best improvements from each worktree
3. **Verify:** Run combined changes through LORO to confirm they stack
4. **Port to production:** Update model.py + retrain_gbt.py with winning changes
5. **Retrain:** Run retrain_gbt.py to rebuild production models

## Files

| File | Who edits it | Purpose |
|------|-------------|---------|
| `train.py` | Agents (in worktrees) | Mutable backtest file — ALL experiments here |
| `model.py` | Us (after agents finish) | Production model — port winning changes |
| `retrain_gbt.py` | Us (after agents finish) | Retrain with new features |
| `analyze_observations.py` | Reference only | Correlation analysis for feature ideas |

## Round Data Available

| Round | GT | Observations | Initial States | Type |
|-------|----|----|----|----|
| R1 | yes | 50 | yes | medium expansion |
| R2 | yes | 50 | yes | high expansion |
| R3 | yes | — | yes | (different dynamics, excluded) |
| R4 | yes | 50 | yes | extinction |
| R5 | yes | 50 | yes | medium expansion |
| R6 | yes | 49 | yes | very high expansion |
| R7 | yes | 50 | yes | extreme expansion |
| R8 | yes | 50 | yes | extinction |
| R9 | yes | 50 | yes | medium |
| R10 | yes | 50 | yes | total extinction |
| R11 | yes | 50 | yes | very high expansion |
| R12 | yes | — | yes | unknown (no obs) |
| R13 | yes | 50 | yes | medium-low |
