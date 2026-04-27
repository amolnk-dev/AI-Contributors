"""Experiment: avg_settlement_spacing feature.

Hypothesis: The average pairwise distance between the 3 nearest settlements
to each cell captures dense-packing vs sparse-distribution dynamics.
- Dense packing → more competition → more ruins, fewer surviving settlements
- Sparse distribution → room to expand → more settlements, fewer ruins

Formula: For each cell, find the 3 nearest settlements, then compute
the average pairwise Manhattan distance between those 3 settlements.

Steps:
1. Compute correlation with GT P(settlement) and P(ruin) across all rounds
2. Run LORO if promising (baseline avg=89.68)
"""

import json
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

from model import (
    _extract_cell_features,
    build_static_prediction,
    fill_unobserved_dynamic,
)
from evaluate import compute_score
from utils import normalize_prediction, load_observations
from dtos import TERRAIN_TO_CLASS, NUM_CLASSES, PROB_FLOOR

import xgboost as xgb

# ── Configuration ──────────────────────────────────────────────

ROUND_IDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
}

ACTIVE_ROUNDS = [1, 2, 4, 5]

L7_STRENGTHS = np.array([1.38, 0.90, 0.44, 0.61, 1.16, 0.0])

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    reg_alpha=0.1,
    reg_lambda=2.0,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    random_state=42,
    verbosity=0,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ── Feature computation ───────────────────────────────────────

def compute_avg_settlement_spacing(y, x, settl_pos, k=3):
    """For cell (y,x), find the k nearest settlements and return
    the average pairwise Manhattan distance between them.

    If fewer than k settlements exist, use all available.
    If fewer than 2 settlements, return 0.
    """
    if len(settl_pos) < 2:
        return 0.0

    # Distances from this cell to all settlements
    dists = [(abs(y - sy) + abs(x - sx), (sy, sx)) for sx, sy in settl_pos]
    dists.sort(key=lambda d: d[0])

    # Take k nearest
    nearest = [d[1] for d in dists[:k]]

    if len(nearest) < 2:
        return 0.0

    # Average pairwise distance between the nearest settlements
    total = 0.0
    count = 0
    for i in range(len(nearest)):
        for j in range(i + 1, len(nearest)):
            sy1, sx1 = nearest[i]
            sy2, sx2 = nearest[j]
            total += abs(sy1 - sy2) + abs(sx1 - sx2)
            count += 1

    return total / count if count > 0 else 0.0


# ── Data loading ──────────────────────────────────────────────

def load_round_data(round_num):
    """Load initial states and GT for a round."""
    with open(os.path.join(DATA_DIR, f"round{round_num}_initial.json")) as f:
        info = json.load(f)
    initial_states = info["initial_states"]
    seeds_count = info["seeds_count"]
    gt_list = []
    for seed in range(seeds_count):
        gt_path = os.path.join(DATA_DIR, f"gt_r{round_num}_seed{seed}.npy")
        gt_list.append(np.load(gt_path))
    return initial_states, gt_list, seeds_count


# ── Step 1: Correlation analysis ──────────────────────────────

def correlation_analysis():
    """Compute correlation of avg_settlement_spacing with GT P(settlement) and P(ruin)."""
    print("=" * 60)
    print("CORRELATION ANALYSIS: avg_settlement_spacing")
    print("=" * 60)

    all_spacing = []
    all_p_settl = []
    all_p_ruin = []
    all_p_forest = []
    all_terrain = []

    for rnum in ACTIVE_ROUNDS:
        initial_states, gt_list, seeds_count = load_round_data(rnum)

        for seed in range(seeds_count):
            state = initial_states[seed]
            grid = state["grid"]
            settlements = state["settlements"]
            gt = gt_list[seed]
            h = len(grid)
            w = len(grid[0]) if h > 0 else 0

            # Settlement positions
            settl_pos = []
            for s in settlements:
                if isinstance(s, dict):
                    settl_pos.append((s.get("x", 0), s.get("y", 0)))
                else:
                    settl_pos.append((s.x, s.y))

            for y in range(h):
                for x in range(w):
                    code = grid[y][x]
                    if code in {10, 5}:  # skip static
                        continue

                    spacing = compute_avg_settlement_spacing(y, x, settl_pos, k=3)
                    all_spacing.append(spacing)
                    all_p_settl.append(gt[y, x, 1])  # P(settlement)
                    all_p_ruin.append(gt[y, x, 3])    # P(ruin)
                    all_p_forest.append(gt[y, x, 4])  # P(forest)
                    all_terrain.append(code)

    spacing = np.array(all_spacing)
    p_settl = np.array(all_p_settl)
    p_ruin = np.array(all_p_ruin)
    p_forest = np.array(all_p_forest)
    terrain = np.array(all_terrain)

    # Overall correlations
    r_settl, pval_settl = stats.pearsonr(spacing, p_settl)
    r_ruin, pval_ruin = stats.pearsonr(spacing, p_ruin)
    r_forest, pval_forest = stats.pearsonr(spacing, p_forest)

    print(f"\nOverall (n={len(spacing)} dynamic cells across {len(ACTIVE_ROUNDS)} rounds x 5 seeds):")
    print(f"  corr(spacing, P(settlement)) = {r_settl:.4f}  (p={pval_settl:.2e})")
    print(f"  corr(spacing, P(ruin))       = {r_ruin:.4f}  (p={pval_ruin:.2e})")
    print(f"  corr(spacing, P(forest))     = {r_forest:.4f}  (p={pval_forest:.2e})")

    # Per-terrain correlations
    for code, name in [(11, "Plains"), (4, "Forest"), (1, "Settlement")]:
        mask = terrain == code
        if mask.sum() < 10:
            continue
        r_s, p_s = stats.pearsonr(spacing[mask], p_settl[mask])
        r_r, p_r = stats.pearsonr(spacing[mask], p_ruin[mask])
        print(f"\n  {name} (code={code}, n={mask.sum()}):")
        print(f"    corr(spacing, P(settlement)) = {r_s:.4f}  (p={p_s:.2e})")
        print(f"    corr(spacing, P(ruin))       = {r_r:.4f}  (p={p_r:.2e})")

    # Binned analysis: mean P(settlement) and P(ruin) by spacing quartile
    print("\n  Binned analysis (spacing quartiles):")
    quartiles = np.percentile(spacing, [0, 25, 50, 75, 100])
    print(f"  Quartile edges: {quartiles}")
    for i in range(4):
        mask = (spacing >= quartiles[i]) & (spacing < quartiles[i + 1] + (1e-6 if i == 3 else 0))
        n = mask.sum()
        if n == 0:
            continue
        print(f"    Q{i+1} (spacing {quartiles[i]:.1f}-{quartiles[i+1]:.1f}, n={n}): "
              f"mean P(settl)={p_settl[mask].mean():.4f}, "
              f"mean P(ruin)={p_ruin[mask].mean():.4f}, "
              f"mean P(forest)={p_forest[mask].mean():.4f}")

    return r_settl, r_ruin


# ── Step 2: LORO evaluation with the new feature ─────────────

def extract_features_with_spacing(grid, settlements):
    """Extract cell features + append avg_settlement_spacing."""
    base_feats, coords = _extract_cell_features(grid, settlements)
    if len(base_feats) == 0:
        return base_feats, coords

    h = len(grid)
    w = len(grid[0]) if h > 0 else 0

    # Settlement positions
    settl_pos = []
    for s in settlements:
        if isinstance(s, dict):
            settl_pos.append((s.get("x", 0), s.get("y", 0)))
        else:
            settl_pos.append((s.x, s.y))

    # Compute spacing for each dynamic cell
    spacing_col = np.zeros((len(coords), 1))
    for i, (y, x) in enumerate(coords):
        spacing_col[i, 0] = compute_avg_settlement_spacing(y, x, settl_pos, k=3)

    augmented = np.hstack([base_feats, spacing_col])
    return augmented, coords


def train_gbt_models(X_data, Y_data):
    """Train terrain-specific XGBoost models."""
    models = {}
    for ttype in ["plains", "forest", "settl"]:
        X = np.array(X_data[ttype])
        Y = np.array(Y_data[ttype])
        terrain_models = []
        for cls in range(NUM_CLASSES):
            m = xgb.XGBRegressor(**XGB_PARAMS)
            m.fit(X, Y[:, cls])
            terrain_models.append(m)
        models[ttype] = terrain_models
    return models


def gbt_predict_with_models(initial_grid, settlements, models_dict, use_spacing=False):
    """Generate GBT predictions using given models."""
    h = len(initial_grid)
    w = len(initial_grid[0]) if h > 0 else 0

    if use_spacing:
        features, coords = extract_features_with_spacing(initial_grid, settlements)
    else:
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
        if code in {11, 0}:
            ttype = "plains"
        elif code == 4:
            ttype = "forest"
        elif code in {1, 2}:
            ttype = "settl"
        else:
            ttype = "plains"

        models = models_dict.get(ttype)
        if models is None:
            continue

        pred_v = np.zeros(NUM_CLASSES)
        for cls in range(NUM_CLASSES):
            pred_v[cls] = models[cls].predict(features[i:i + 1])[0]
        tensor[y, x] = np.maximum(pred_v, PROB_FLOOR)
        tensor[y, x] /= tensor[y, x].sum()

    return tensor


def apply_l7_correction(tensor, initial_grid, all_observations):
    """Apply Layer 7 global observation ratio correction."""
    if not all_observations:
        return tensor

    obs_cls = np.zeros(NUM_CLASSES)
    obs_total = 0
    for obs in all_observations:
        for row in obs.get("grid", []):
            for code in row:
                if code not in {10, 5}:
                    obs_cls[TERRAIN_TO_CLASS.get(code, 0)] += 1
                    obs_total += 1

    if obs_total <= 100:
        return tensor

    obs_freq = obs_cls / obs_total
    h, w, _ = tensor.shape
    model_avg = np.zeros(NUM_CLASSES)
    m_count = 0
    for y in range(h):
        for x in range(w):
            if initial_grid[y][x] not in {10, 5}:
                model_avg += tensor[y, x]
                m_count += 1
    if m_count > 0:
        model_avg /= m_count
        ratio = obs_freq / np.maximum(model_avg, 1e-6)
        adj = 1.0 + L7_STRENGTHS * (ratio - 1.0)
        for y in range(h):
            for x in range(w):
                if initial_grid[y][x] in {10, 5}:
                    continue
                tensor[y, x] *= adj
                tensor[y, x] = np.maximum(tensor[y, x], PROB_FLOOR)
                tensor[y, x] /= tensor[y, x].sum()

    return tensor


def build_prediction_with_blend(initial_grid, settlements, gbt_models, blend_weight, all_observations, use_spacing=False):
    """Build full prediction: heuristic -> GBT blend -> L7 correction."""
    tensor = build_static_prediction(initial_grid)
    tensor = fill_unobserved_dynamic(tensor, initial_grid, settlements, [], 0)

    gbt_pred = gbt_predict_with_models(initial_grid, settlements, gbt_models, use_spacing=use_spacing)
    if gbt_pred is not None:
        h, w, _ = tensor.shape
        for y in range(h):
            for x in range(w):
                code = initial_grid[y][x]
                if code in {10, 5}:
                    continue
                tensor[y, x] = (1 - blend_weight) * tensor[y, x] + blend_weight * gbt_pred[y, x]

    tensor = apply_l7_correction(tensor, initial_grid, all_observations)
    tensor = normalize_prediction(tensor)
    return tensor


def extract_training_data(round_num, use_spacing=False):
    """Extract features and targets from one round's GT."""
    initial_states, gt_list, seeds_count = load_round_data(round_num)

    X_data = {"plains": [], "forest": [], "settl": []}
    Y_data = {"plains": [], "forest": [], "settl": []}

    for seed in range(seeds_count):
        state = initial_states[seed]
        grid = state["grid"]
        settlements = state["settlements"]
        gt = gt_list[seed]

        if use_spacing:
            feats, coords = extract_features_with_spacing(grid, settlements)
        else:
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

    return X_data, Y_data


def loro_evaluate(use_spacing=False, blend_weight=0.30, label=""):
    """Leave-One-Round-Out cross-validation."""
    print(f"\n{'='*60}")
    print(f"LORO: {label} (blend={blend_weight})")
    print(f"{'='*60}")

    # Pre-load all round data
    round_data = {}
    for rn in ACTIVE_ROUNDS:
        initial_states, gt_list, seeds_count = load_round_data(rn)
        obs = load_observations(ROUND_IDS[rn])
        round_data[rn] = {
            "initial_states": initial_states,
            "gt_list": gt_list,
            "seeds_count": seeds_count,
            "observations": obs,
        }

    round_scores = {}
    for test_round in ACTIVE_ROUNDS:
        train_rounds = [r for r in ACTIVE_ROUNDS if r != test_round]

        # Collect training data
        X_all = {"plains": [], "forest": [], "settl": []}
        Y_all = {"plains": [], "forest": [], "settl": []}

        for tr in train_rounds:
            X_r, Y_r = extract_training_data(tr, use_spacing=use_spacing)
            for ttype in ["plains", "forest", "settl"]:
                X_all[ttype].extend(X_r[ttype])
                Y_all[ttype].extend(Y_r[ttype])

        n_train = sum(len(X_all[t]) for t in X_all)
        n_features = len(X_all["plains"][0]) if X_all["plains"] else 0
        print(f"  Test=R{test_round}, Train={train_rounds} ({n_train} samples, {n_features} features)")

        # Train
        gbt_models = train_gbt_models(X_all, Y_all)

        # Evaluate
        test_data = round_data[test_round]
        seed_scores = []
        for seed in range(test_data["seeds_count"]):
            state = test_data["initial_states"][seed]
            grid = state["grid"]
            settlements = state["settlements"]
            gt = test_data["gt_list"][seed]
            obs = test_data["observations"]
            seed_obs = [o for o in obs if o.get("seed_index") == seed]

            pred = build_prediction_with_blend(
                grid, settlements, gbt_models, blend_weight, seed_obs,
                use_spacing=use_spacing,
            )
            score = compute_score(pred, gt)
            seed_scores.append(score)

        avg = np.mean(seed_scores)
        round_scores[test_round] = avg
        print(f"    R{test_round}: avg={avg:.2f}  seeds=[{', '.join(f'{s:.2f}' for s in seed_scores)}]")

    overall = np.mean(list(round_scores.values()))
    print(f"  LORO avg: {overall:.4f}")
    return round_scores, overall


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    t0 = time.time()

    # Step 1: Correlation analysis
    r_settl, r_ruin = correlation_analysis()

    # Step 2: LORO — always run to get definitive answer
    print("\n\n")

    # Baseline (no spacing feature)
    baseline_scores, baseline_avg = loro_evaluate(
        use_spacing=False, blend_weight=0.30, label="Baseline (30 features)")

    # With spacing feature
    spacing_scores, spacing_avg = loro_evaluate(
        use_spacing=True, blend_weight=0.30, label="With avg_settlement_spacing (31 features)")

    # Summary
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Approach':<45} {'R1':>7} {'R2':>7} {'R4':>7} {'R5':>7} {'AVG':>7}")
    print("-" * 80)
    for label, scores, avg in [
        ("Baseline (30 features)", baseline_scores, baseline_avg),
        ("+ avg_settlement_spacing (31 features)", spacing_scores, spacing_avg),
    ]:
        print(f"{label:<45} {scores.get(1, 0):>7.2f} {scores.get(2, 0):>7.2f} "
              f"{scores.get(4, 0):>7.2f} {scores.get(5, 0):>7.2f} {avg:>7.2f}")

    delta = spacing_avg - baseline_avg
    print(f"\nDelta: {delta:+.4f}")
    print(f"Total time: {time.time() - t0:.1f}s")
