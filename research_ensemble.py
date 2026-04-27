"""RESEARCH ONLY — Multi-model ensemble experiments.

Tests ensembling diverse model configurations in 6-fold LORO:
1. Blend ensemble: avg of blend=0.20 and blend=0.50
2. L7 ensemble: avg of no-L7 and safe-L7
3. Triple ensemble: avg of 3 diverse configs

Hypothesis: different configs excel on different rounds; ensembling
captures the best of each.
"""

import json
import os
import sys
import time
import numpy as np
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

import xgboost as xgb
from model import (
    _extract_cell_features,
    build_static_prediction,
    fill_unobserved_dynamic,
)
from evaluate import compute_score
from utils import normalize_prediction, load_observations
from dtos import TERRAIN_TO_CLASS, NUM_CLASSES, PROB_FLOOR

# ──────────────────────────────────────────────────────────────
# Round data
# ──────────────────────────────────────────────────────────────

ROUNDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    6: "ae78003a-4efe-425a-881a-d16a39bca0ad",
    7: "36e581f1-73f8-453f-ab98-cbe3052b701b",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

XGB_HPARAMS = {
    "plains": dict(n_estimators=200, max_depth=4, learning_rate=0.1,
                   reg_alpha=0.1, reg_lambda=2.0, subsample=0.9,
                   colsample_bytree=0.9, min_child_weight=3),
    "forest": dict(n_estimators=200, max_depth=4, learning_rate=0.1,
                   reg_alpha=0.1, reg_lambda=2.0, subsample=0.9,
                   colsample_bytree=0.9, min_child_weight=3),
    "settl":  dict(n_estimators=200, max_depth=4, learning_rate=0.1,
                   reg_alpha=0.1, reg_lambda=2.0, subsample=0.9,
                   colsample_bytree=0.9, min_child_weight=3),
}

# Safe L7 settings (from task description)
SAFE_L7_STRENGTHS = np.array([1.20, 0.80, 0.0, 0.0, 1.30, 0.0])
SAFE_L7_MIN_OBS = np.array([200, 50, 30, 20, 100, 0])
SAFE_L7_CLAMP_MIN = 0.80
SAFE_L7_CLAMP_MAX = 1.25


def load_round_data(round_num):
    with open(os.path.join(DATA_DIR, f"round{round_num}_initial.json")) as f:
        info = json.load(f)
    initial_states = info["initial_states"]
    gts = []
    for seed in range(5):
        gts.append(np.load(os.path.join(DATA_DIR, f"gt_r{round_num}_seed{seed}.npy")))
    return initial_states, gts


def train_gbt_models(train_rounds):
    """Train terrain-specific XGBoost on given rounds."""
    X_data = {"plains": [], "forest": [], "settl": []}
    Y_data = {"plains": [], "forest": [], "settl": []}

    for rnum in train_rounds:
        initial_states, gts = load_round_data(rnum)
        for seed in range(5):
            grid = initial_states[seed]["grid"]
            settlements = initial_states[seed]["settlements"]
            gt = gts[seed]
            feats, coords = _extract_cell_features(grid, settlements)
            targets = np.array([gt[y, x] for y, x in coords])
            for i, (y, x) in enumerate(coords):
                code = grid[y][x]
                if code in {11, 0}:
                    X_data["plains"].append(feats[i])
                    Y_data["plains"].append(targets[i])
                elif code == 4:
                    X_data["forest"].append(feats[i])
                    Y_data["forest"].append(targets[i])
                elif code in {1, 2}:
                    X_data["settl"].append(feats[i])
                    Y_data["settl"].append(targets[i])

    models = {}
    for terrain_type in ["plains", "forest", "settl"]:
        X = np.array(X_data[terrain_type])
        Y = np.array(Y_data[terrain_type])
        hp = XGB_HPARAMS[terrain_type]
        terrain_models = []
        for cls in range(6):
            m = xgb.XGBRegressor(
                n_estimators=hp["n_estimators"],
                max_depth=hp["max_depth"],
                learning_rate=hp["learning_rate"],
                reg_alpha=hp["reg_alpha"],
                reg_lambda=hp["reg_lambda"],
                subsample=hp["subsample"],
                colsample_bytree=hp["colsample_bytree"],
                min_child_weight=hp["min_child_weight"],
                random_state=42, verbosity=0,
            )
            m.fit(X, Y[:, cls])
            terrain_models.append(m)
        models[terrain_type] = terrain_models
    return models


def gbt_predict_with_models(models_dict, initial_grid, settlements):
    h = len(initial_grid)
    w = len(initial_grid[0]) if h > 0 else 0
    features, coords = _extract_cell_features(initial_grid, settlements)
    if len(features) == 0:
        return None
    tensor = np.zeros((h, w, NUM_CLASSES))
    for y in range(h):
        for x in range(w):
            if initial_grid[y][x] == 10:
                tensor[y, x] = [1, 0, 0, 0, 0, 0]
            elif initial_grid[y][x] == 5:
                tensor[y, x] = [0, 0, 0, 0, 0, 1]
    for i, (y, x) in enumerate(coords):
        code = initial_grid[y][x]
        ttype = "plains" if code in {11, 0} else ("forest" if code == 4 else ("settl" if code in {1, 2} else "plains"))
        models = models_dict.get(ttype)
        if models is None:
            continue
        pred_v = np.zeros(NUM_CLASSES)
        for cls in range(NUM_CLASSES):
            pred_v[cls] = models[cls].predict(features[i:i + 1])[0]
        tensor[y, x] = np.maximum(pred_v, PROB_FLOOR)
        tensor[y, x] /= tensor[y, x].sum()
    return tensor


def build_single_prediction(grid, settlements, gbt_models, blend_weight,
                             all_observations, use_l7=False,
                             l7_strengths=None, l7_min_obs=None,
                             l7_clamp_min=0.80, l7_clamp_max=1.25):
    """Build a single prediction tensor with given config.

    Returns the raw tensor BEFORE final normalization (so we can ensemble raw).
    """
    # Layer 1+3: Static + context priors
    tensor = build_static_prediction(grid)
    tensor = fill_unobserved_dynamic(tensor, grid, settlements, [], 0)

    # Layer 6: GBT blend
    gbt_pred = gbt_predict_with_models(gbt_models, grid, settlements)
    if gbt_pred is not None:
        h, w, _ = tensor.shape
        for y in range(h):
            for x in range(w):
                if grid[y][x] not in {10, 5}:
                    tensor[y, x] = (1 - blend_weight) * tensor[y, x] + blend_weight * gbt_pred[y, x]

    # Layer 7: Observation ratio correction (optional)
    if use_l7 and all_observations:
        if l7_strengths is None:
            l7_strengths = SAFE_L7_STRENGTHS
        if l7_min_obs is None:
            l7_min_obs = SAFE_L7_MIN_OBS

        obs_cls = np.zeros(NUM_CLASSES)
        obs_total = 0
        for obs in all_observations:
            for row in obs.get("grid", []):
                for code in row:
                    if code not in {10, 5}:
                        obs_cls[TERRAIN_TO_CLASS.get(code, 0)] += 1
                        obs_total += 1
        if obs_total > 100:
            obs_freq = obs_cls / obs_total
            h, w, _ = tensor.shape
            model_avg = np.zeros(NUM_CLASSES)
            m_count = 0
            for y in range(h):
                for x in range(w):
                    if grid[y][x] not in {10, 5}:
                        model_avg += tensor[y, x]
                        m_count += 1
            if m_count > 0:
                model_avg /= m_count
                ratio = obs_freq / np.maximum(model_avg, 1e-6)
                strengths = l7_strengths.copy()
                for c in range(NUM_CLASSES):
                    if obs_cls[c] < l7_min_obs[c]:
                        strengths[c] = 0.0
                adj = 1.0 + strengths * (ratio - 1.0)
                adj = np.clip(adj, l7_clamp_min, l7_clamp_max)
                for y in range(h):
                    for x in range(w):
                        if grid[y][x] in {10, 5}:
                            continue
                        tensor[y, x] *= adj
                        tensor[y, x] = np.maximum(tensor[y, x], PROB_FLOOR)
                        tensor[y, x] /= tensor[y, x].sum()

    return tensor


def evaluate_single_config(test_round_num, gbt_models, blend_weight, use_l7,
                           l7_strengths=None, l7_min_obs=None,
                           l7_clamp_min=0.80, l7_clamp_max=1.25):
    """Evaluate a single config on a round, returning (mean_score, per_seed_scores)."""
    round_id = ROUNDS[test_round_num]
    initial_states, gts = load_round_data(test_round_num)
    all_observations = load_observations(round_id)

    scores = []
    for seed in range(5):
        grid = initial_states[seed]["grid"]
        settlements = initial_states[seed]["settlements"]
        gt = gts[seed]

        tensor = build_single_prediction(
            grid, settlements, gbt_models, blend_weight,
            all_observations, use_l7=use_l7,
            l7_strengths=l7_strengths, l7_min_obs=l7_min_obs,
            l7_clamp_min=l7_clamp_min, l7_clamp_max=l7_clamp_max,
        )
        tensor = normalize_prediction(tensor)
        score = compute_score(tensor, gt)
        scores.append(score)

    return np.mean(scores), scores


def evaluate_ensemble(test_round_num, gbt_models, configs):
    """Evaluate an ensemble of configs. Each config is a dict of kwargs for build_single_prediction.

    Averages the raw tensors before normalization.
    """
    round_id = ROUNDS[test_round_num]
    initial_states, gts = load_round_data(test_round_num)
    all_observations = load_observations(round_id)

    scores = []
    for seed in range(5):
        grid = initial_states[seed]["grid"]
        settlements = initial_states[seed]["settlements"]
        gt = gts[seed]

        # Build each component prediction
        preds = []
        for cfg in configs:
            tensor = build_single_prediction(
                grid, settlements, gbt_models,
                blend_weight=cfg["blend_weight"],
                all_observations=all_observations,
                use_l7=cfg.get("use_l7", False),
                l7_strengths=cfg.get("l7_strengths"),
                l7_min_obs=cfg.get("l7_min_obs"),
                l7_clamp_min=cfg.get("l7_clamp_min", 0.80),
                l7_clamp_max=cfg.get("l7_clamp_max", 1.25),
            )
            preds.append(tensor)

        # Average the tensors
        ensemble_pred = sum(preds) / len(preds)

        # Normalize and score
        ensemble_pred = normalize_prediction(ensemble_pred)
        score = compute_score(ensemble_pred, gt)
        scores.append(score)

    return np.mean(scores), scores


def main():
    start = time.time()
    test_rounds = [1, 2, 4, 5, 6, 7]

    # ──────────────────────────────────────────────────────────────
    # First: run the baseline (blend=0.35, safe L7) for comparison
    # ──────────────────────────────────────────────────────────────
    print("=" * 80)
    print("BASELINE: blend=0.35 + safe L7")
    print("=" * 80)
    baseline_scores = {}
    for held_out in test_rounds:
        train_on = [r for r in test_rounds if r != held_out]
        gbt_models = train_gbt_models(train_on)
        score, _ = evaluate_single_config(
            held_out, gbt_models, blend_weight=0.35, use_l7=True,
            l7_strengths=SAFE_L7_STRENGTHS, l7_min_obs=SAFE_L7_MIN_OBS,
            l7_clamp_min=SAFE_L7_CLAMP_MIN, l7_clamp_max=SAFE_L7_CLAMP_MAX,
        )
        baseline_scores[held_out] = score
        print(f"  R{held_out}: {score:.2f}", flush=True)
    baseline_avg = np.mean(list(baseline_scores.values()))
    print(f"  AVG: {baseline_avg:.2f}")
    print()

    # Also get no-L7 baseline for reference
    print("=" * 80)
    print("BASELINE (no L7): blend=0.35, no L7")
    print("=" * 80)
    no_l7_scores = {}
    for held_out in test_rounds:
        train_on = [r for r in test_rounds if r != held_out]
        gbt_models = train_gbt_models(train_on)
        score, _ = evaluate_single_config(
            held_out, gbt_models, blend_weight=0.35, use_l7=False,
        )
        no_l7_scores[held_out] = score
        print(f"  R{held_out}: {score:.2f}", flush=True)
    no_l7_avg = np.mean(list(no_l7_scores.values()))
    print(f"  AVG: {no_l7_avg:.2f}")
    print()

    # ──────────────────────────────────────────────────────────────
    # Experiment 1: Blend ensemble (avg of blend=0.20 and blend=0.50)
    # ──────────────────────────────────────────────────────────────
    print("=" * 80)
    print("EXPERIMENT 1: Blend Ensemble — avg(blend=0.20, blend=0.50)")
    print("  Both with safe L7")
    print("=" * 80)
    blend_ens_scores = {}
    for held_out in test_rounds:
        train_on = [r for r in test_rounds if r != held_out]
        gbt_models = train_gbt_models(train_on)
        configs = [
            {"blend_weight": 0.20, "use_l7": True,
             "l7_strengths": SAFE_L7_STRENGTHS, "l7_min_obs": SAFE_L7_MIN_OBS,
             "l7_clamp_min": SAFE_L7_CLAMP_MIN, "l7_clamp_max": SAFE_L7_CLAMP_MAX},
            {"blend_weight": 0.50, "use_l7": True,
             "l7_strengths": SAFE_L7_STRENGTHS, "l7_min_obs": SAFE_L7_MIN_OBS,
             "l7_clamp_min": SAFE_L7_CLAMP_MIN, "l7_clamp_max": SAFE_L7_CLAMP_MAX},
        ]
        score, _ = evaluate_ensemble(held_out, gbt_models, configs)
        blend_ens_scores[held_out] = score
        delta = score - baseline_scores[held_out]
        print(f"  R{held_out}: {score:.2f} (delta {delta:+.2f})", flush=True)
    blend_ens_avg = np.mean(list(blend_ens_scores.values()))
    print(f"  AVG: {blend_ens_avg:.2f} (delta {blend_ens_avg - baseline_avg:+.2f})")
    print()

    # ──────────────────────────────────────────────────────────────
    # Experiment 2: L7 ensemble (avg of no-L7 and safe-L7)
    # ──────────────────────────────────────────────────────────────
    print("=" * 80)
    print("EXPERIMENT 2: L7 Ensemble — avg(no_L7 + blend=0.35, safe_L7 + blend=0.35)")
    print("=" * 80)
    l7_ens_scores = {}
    for held_out in test_rounds:
        train_on = [r for r in test_rounds if r != held_out]
        gbt_models = train_gbt_models(train_on)
        configs = [
            {"blend_weight": 0.35, "use_l7": False},
            {"blend_weight": 0.35, "use_l7": True,
             "l7_strengths": SAFE_L7_STRENGTHS, "l7_min_obs": SAFE_L7_MIN_OBS,
             "l7_clamp_min": SAFE_L7_CLAMP_MIN, "l7_clamp_max": SAFE_L7_CLAMP_MAX},
        ]
        score, _ = evaluate_ensemble(held_out, gbt_models, configs)
        l7_ens_scores[held_out] = score
        delta = score - baseline_scores[held_out]
        print(f"  R{held_out}: {score:.2f} (delta {delta:+.2f})", flush=True)
    l7_ens_avg = np.mean(list(l7_ens_scores.values()))
    print(f"  AVG: {l7_ens_avg:.2f} (delta {l7_ens_avg - baseline_avg:+.2f})")
    print()

    # ──────────────────────────────────────────────────────────────
    # Experiment 3: Triple ensemble
    # avg(no_L7+blend=0.25, safe_L7+blend=0.35, safe_L7+blend=0.45)
    # ──────────────────────────────────────────────────────────────
    print("=" * 80)
    print("EXPERIMENT 3: Triple Ensemble")
    print("  avg(no_L7+blend=0.25, safe_L7+blend=0.35, safe_L7+blend=0.45)")
    print("=" * 80)
    triple_ens_scores = {}
    for held_out in test_rounds:
        train_on = [r for r in test_rounds if r != held_out]
        gbt_models = train_gbt_models(train_on)
        configs = [
            {"blend_weight": 0.25, "use_l7": False},
            {"blend_weight": 0.35, "use_l7": True,
             "l7_strengths": SAFE_L7_STRENGTHS, "l7_min_obs": SAFE_L7_MIN_OBS,
             "l7_clamp_min": SAFE_L7_CLAMP_MIN, "l7_clamp_max": SAFE_L7_CLAMP_MAX},
            {"blend_weight": 0.45, "use_l7": True,
             "l7_strengths": SAFE_L7_STRENGTHS, "l7_min_obs": SAFE_L7_MIN_OBS,
             "l7_clamp_min": SAFE_L7_CLAMP_MIN, "l7_clamp_max": SAFE_L7_CLAMP_MAX},
        ]
        score, _ = evaluate_ensemble(held_out, gbt_models, configs)
        triple_ens_scores[held_out] = score
        delta = score - baseline_scores[held_out]
        print(f"  R{held_out}: {score:.2f} (delta {delta:+.2f})", flush=True)
    triple_ens_avg = np.mean(list(triple_ens_scores.values()))
    print(f"  AVG: {triple_ens_avg:.2f} (delta {triple_ens_avg - baseline_avg:+.2f})")
    print()

    # ──────────────────────────────────────────────────────────────
    # Summary table
    # ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    header = f"{'Config':<35}"
    for r in test_rounds:
        header += f"{'R'+str(r):>8}"
    header += f"{'AVG':>8}  {'Delta':>7}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Baseline (blend=0.35+safeL7)", baseline_scores, baseline_avg),
        ("No-L7 baseline (blend=0.35)", no_l7_scores, no_l7_avg),
        ("Exp1: Blend ens (0.20+0.50)", blend_ens_scores, blend_ens_avg),
        ("Exp2: L7 ens (noL7+safeL7)", l7_ens_scores, l7_ens_avg),
        ("Exp3: Triple ens", triple_ens_scores, triple_ens_avg),
    ]

    for label, scores, avg in rows:
        row = f"{label:<35}"
        for r in test_rounds:
            row += f"{scores[r]:>8.2f}"
        delta = avg - baseline_avg
        row += f"{avg:>8.2f}  {delta:>+7.2f}"
        print(row)

    print()
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
