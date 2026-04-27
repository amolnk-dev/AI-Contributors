# READTHIS — Astar Island Parameter Sync Guide

## The Problem We Found (2026-03-22)

train.py, model.py, and retrain_gbt.py each had DIFFERENT parameter values.
LORO validates against train.py's pipeline, but the actual R22 submission
runs through model.py. Result: our submission was using stale, unvalidated
parameters. Fixing this sync improved WAVG from 89.90 to 90.16.

## The Three Files That Must Stay In Sync

```
train.py        — SOURCE OF TRUTH. LORO backtests against real GT maps.
model.py        — INFERENCE PATH. build_prediction() generates submissions.
retrain_gbt.py  — GBT TRAINING. Produces data/gbt_models.pkl.
```

### What must match between train.py and model.py:

| Parameter | train.py location | model.py location |
|---|---|---|
| BLEND_TERRAIN | Top-level constant | build_prediction() L6 section |
| EMP_BLEND | evaluate_loro() L6.7 section | build_prediction() L6.7 `EMPIRICAL_BLEND_WEIGHT` |
| L7 strengths | ROUND_TYPE_L7 dict | build_prediction() L7 `cls_strength` / `ROUND_TYPE_L7` |
| L7 clamps (adj_min/adj_max) | ROUND_TYPE_L7 dict | build_prediction() L7 `adj_min` / `adj_max` |
| L7_MIN_OBS | Top-level constant | build_prediction() L7 `min_obs` / `L7_MIN_OBS` |
| Round-type detection | classify_round_type() | classify_round_type() — must exist in model.py |
| Round-type thresholds | ROUND_TYPE_THRESHOLDS | ROUND_TYPE_THRESHOLDS |

### What must match between train.py and retrain_gbt.py:

| Parameter | train.py | retrain_gbt.py |
|---|---|---|
| XGB_HPARAMS | Per-terrain dict | Per-terrain dict (must be identical) |
| ROUNDS dict | Round IDs | Round IDs (same rounds included/excluded) |
| ROUND_WEIGHTS | Weight dict | Weight dict |
| Feature pipeline | _extract_cell_features + compute_obs_stats + compute_cell_obs_features | Same functions from model.py |

## Sync Procedure (do this every time you change parameters)

1. **Edit train.py** — change parameters
2. **Run LORO** — `python train.py` — verify val_metric improves
3. **Sync to model.py** — copy the exact same parameter values into model.py's build_prediction()
4. **Sync to retrain_gbt.py** — if XGB_HPARAMS or ROUNDS changed
5. **Retrain GBT** — `python retrain_gbt.py` — regenerates data/gbt_models.pkl
6. **Submit** — `ASTAR_TOKEN=... python run.py --resume`

## Architecture: Why Three Files?

```
train.py ──── LORO evaluation (offline backtesting, ~8 min)
  │            Uses its own prediction pipeline with tunable params
  │            This is what we optimize against
  │
model.py ──── Production inference (called by run.py for submissions)
  │            build_prediction() must mirror train.py's pipeline
  │            Also contains shared functions: _extract_cell_features,
  │            compute_obs_stats, compute_cell_obs_features, etc.
  │
retrain_gbt.py ── Trains XGBoost models (saved to gbt_models.pkl)
                   Must use same XGB_HPARAMS as train.py
                   Must train on same ROUNDS as train.py
```

train.py imports functions FROM model.py (feature extractors, static prediction).
model.py does NOT import from train.py (no circular dependency).
So parameters in train.py must be manually copied to model.py when changed.

## The Swarm VMs

The 952 GCP VMs run autoresearch_swarm.py which edits train.py via Gemini.
Their LORO is 1:1 with local (same data, same code, same evaluation).
BUT: VM improvements only affect train.py — they don't auto-sync to model.py.

To use a VM's improvement:
1. Pull the VM's train.py: `gcloud compute scp VM:/tmp/astar/train.py .`
2. Diff against local: `diff train.py train.py.from_vm`
3. Apply parameter changes to BOTH train.py AND model.py
4. Retrain GBT + resubmit

## Current Validated Parameters (2026-03-22, WAVG=90.16)

```python
# Layer 6: GBT blend
BLEND_TERRAIN = {"plains": 1.00, "forest": 1.00, "settl": 1.00}

# Layer 6.7: Empirical tables (round-type-aware)
EMP_BLEND = 0.35  # expansion
EMP_BLEND = 0.50  # normal/extinction

# Layer 7: Round-type-adaptive L7
ROUND_TYPE_THRESHOLDS = {"extinction": 0.04, "expansion": 0.12}
# See train.py ROUND_TYPE_L7 dict for full per-type strengths/clamps

# XGB
n_estimators=600, max_depth=5, lr=0.08, reg_alpha=0.1,
reg_lambda=2.0, subsample=0.9, colsample_bytree=0.55, min_child_weight=3

# Rounds: R1-R19, R21 (R3, R12, R20 excluded)
```
