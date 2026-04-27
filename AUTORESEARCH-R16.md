# Autoresearch Session: R16 Era (2026-03-21)

## Baseline

- **WAVG: 89.39** (13-fold LORO, R1-R15 excluding R12)
- **Features: 76** (30 cell + 23 obs stats + 23 per-cell obs)
- **XGB: n_est=600, depth=5, lr=0.08, colsample=0.55**
- **Blend: plains=1.00, forest=1.00, settl=1.00** (in train.py LORO)
- **Production blend: plains=0.90, forest=0.75, settl=0.80** (in model.py)

### Per-Round Scores (weakest first)
```
R7  = 72.54  ← BIGGEST OPPORTUNITY (+17 pts below mean)
R1  = 85.55
R14 = 85.67
R5  = 86.52
R6  = 87.56
R11 = 87.74
R2  = 92.14
R15 = 92.42
R10 = 92.99
R9  = 93.08
R13 = 93.89
R4  = 94.26
R8  = 95.26
```

**Pattern: Strong on extinction, weak on expansion.**

## Branch Structure

```
main (WAVG=89.39 — protected production baseline)
  └── autoresearch/r16 (working branch)
        ├── worktree-expansion/ → Agent A (R7/expansion specialist)
        ├── worktree-features/  → Agent B (feature engineering)
        └── worktree-params/    → Agent C (parameter tuning)
```

## The Three Agents

### Agent A: R7/Expansion Specialist

**Goal:** Fix R7=72.54 and improve R1, R5, R6, R11 (all expansion rounds).

**Why weak:** High expansion rounds have settlements spreading far from initial positions.
Our distance tables were calibrated on ALL rounds including extinction, biasing toward less expansion.

**Ideas:**
- Round-type detection from obs_stats (settl_rate > 0.15 = expansion)
- Per-round-type distance tables (expansion vs extinction calibration)
- Adaptive blend weights per round type
- Non-linear distance decay for expansion scenarios
- Expansion pressure gradient (more settlements nearby = higher P(settlement))

### Agent B: Feature Engineering

**Goal:** Add new XGBoost features that capture hidden dynamics.

**Ideas:**
- Terrain diversity in radius 3/5
- Settlement viability: food access / competition ratio
- Port BFS distance over passable terrain
- Faction diversity per cell radius
- Defense variance (raiding indicator)
- Wealth depletion rate
- Forest reclamation rate from obs grids

### Agent C: Parameter Tuner

**Goal:** Systematic grid search of existing knobs.

**Sweeps:**
- EMP_BLEND: [0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55]
- Per-terrain blend: ±0.05 on each
- L7 per-class strengths: ±0.2
- L7 clamp bounds: tighten/widen
- XGB: colsample [0.45-0.65], subsample [0.80-0.95]

## Monitoring Commands

### Check agent progress
```bash
# Dashboard
cd tasks/astar-island && python orchestrate.py --status

# Per-agent logs
tail -f ../worktree-expansion/tasks/astar-island/run.log
tail -f ../worktree-features/tasks/astar-island/run.log
tail -f ../worktree-params/tasks/astar-island/run.log

# Per-agent git history (kept experiments)
cd ../worktree-expansion && git log --oneline -10
cd ../worktree-features && git log --oneline -10
cd ../worktree-params && git log --oneline -10
```

### GCP fleet
```bash
# Fleet health + costs
scripts/gcp/status.sh

# Experiment progress
scripts/gcp/fleet-experiment.sh check

# Collect results
scripts/gcp/fleet-experiment.sh collect

# Tail VM log
gcloud compute ssh ainm-astar-sweep --zone europe-west4-a --command='tail -50 /tmp/astar/run.log'
```

## Orchestrator

`orchestrate.py` runs periodically to:
1. Check each worktree for new improvements
2. Test if improvements stack (merge + LORO)
3. Log results to `orchestrator.log`
4. Print dashboard

```bash
# Run orchestrator check
python orchestrate.py --status

# Run full stacking test
python orchestrate.py --stack

# Final integration
python orchestrate.py --final
```

## Integration Checklist

When agents finish:

1. [ ] Run `python orchestrate.py --final` — test all combinations
2. [ ] Pick best combination
3. [ ] Apply best train.py parameters to `autoresearch/r16`
4. [ ] If features changed: port to model.py
5. [ ] Run `python retrain_gbt.py`
6. [ ] Run LORO — confirm WAVG > 89.39
7. [ ] `python run.py --dry-run` — verify predictions valid
8. [ ] Merge `autoresearch/r16` → main
9. [ ] Submit for next round

## Key Learnings from Previous Autoresearch

1. **Feature engineering > parameter tuning** (+0.85 vs +0.32)
2. **Blend weights must co-evolve with features** (plains 0.55 → 0.90 after adding features)
3. **Use worktrees** — two agents editing same file = race conditions
4. **colsample scales inversely with feature count** (76 features → 0.55 optimal)
5. **Per-cell obs_settl_rate** (r=0.41 with GT) was biggest single lever
