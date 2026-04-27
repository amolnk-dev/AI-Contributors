"""Experiment: bottleneck_score feature.

How constrained is a cell's connectivity? Cells in narrow passages between
ocean/mountains behave differently from open plains.

bottleneck_score = passable_neighbors_r2 / 12
  where passable_neighbors_r2 = count of cells within Manhattan distance 2
  (excluding center) that are NOT ocean (10) and NOT mountain (5).
  Max neighbors at r=2 = 12. Values near 0 = extreme bottleneck, near 1 = open.

Steps:
  1. Compute correlation of bottleneck_score with GT across all 5 seeds × available rounds
  2. If promising, run LORO CV with the new feature added to XGBoost
"""

import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

import xgboost as xgb
from model import (
    _extract_cell_features,
    build_static_prediction,
    fill_unobserved_dynamic,
    GBT_BLEND_WEIGHT,
)
from evaluate import compute_score
from utils import normalize_prediction, load_observations
from dtos import TERRAIN_TO_CLASS, NUM_CLASSES, PROB_FLOOR

# Round info
ROUNDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
}

L7_STRENGTHS = np.array([1.38, 0.90, 0.44, 0.61, 1.16, 0.0])

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def compute_bottleneck_grid(initial_grid):
    """Compute bottleneck_score for every cell on the grid.

    bottleneck_score = (passable neighbors within Manhattan distance 2) / 12.
    Passable = not ocean (10) and not mountain (5).
    Returns H x W array of floats in [0, 1].
    """
    h = len(initial_grid)
    w = len(initial_grid[0]) if h > 0 else 0
    grid_arr = np.array(initial_grid)
    passable = ((grid_arr != 10) & (grid_arr != 5)).astype(np.float64)

    result = np.zeros((h, w), dtype=np.float64)
    # Manhattan distance <= 2 offsets, excluding center
    offsets = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if abs(dy) + abs(dx) <= 2 and not (dy == 0 and dx == 0):
                offsets.append((dy, dx))
    # Should be 12 offsets

    for dy, dx in offsets:
        # Shift the passable mask
        if dy >= 0:
            src_y = slice(0, h - dy) if dy > 0 else slice(0, h)
            dst_y = slice(dy, h) if dy > 0 else slice(0, h)
        else:
            src_y = slice(-dy, h)
            dst_y = slice(0, h + dy)
        if dx >= 0:
            src_x = slice(0, w - dx) if dx > 0 else slice(0, w)
            dst_x = slice(dx, w) if dx > 0 else slice(0, w)
        else:
            src_x = slice(-dx, w)
            dst_x = slice(0, w + dx)
        result[dst_y, dst_x] += passable[src_y, src_x]

    return result / 12.0


def extract_features_with_bottleneck(initial_grid, settlements):
    """Extract standard features + bottleneck_score appended."""
    base_feats, coords = _extract_cell_features(initial_grid, settlements)
    if len(base_feats) == 0:
        return base_feats, coords

    bn_grid = compute_bottleneck_grid(initial_grid)
    bn_values = np.array([bn_grid[y, x] for y, x in coords]).reshape(-1, 1)
    augmented = np.hstack([base_feats, bn_values])
    return augmented, coords


def load_round_data(round_num):
    """Load initial states and GT for a round."""
    with open(os.path.join(DATA_DIR, f"round{round_num}_initial.json")) as f:
        info = json.load(f)
    initial_states = info["initial_states"]
    gts = []
    for seed in range(5):
        gt_path = os.path.join(DATA_DIR, f"gt_r{round_num}_seed{seed}.npy")
        gts.append(np.load(gt_path))
    return initial_states, gts


# ──────────────────────────────────────────────────────────────
# Step 1: Correlation analysis
# ──────────────────────────────────────────────────────────────

def analyze_correlation():
    """Compute correlation of bottleneck_score with GT class probabilities."""
    print("=" * 60)
    print("STEP 1: Correlation of bottleneck_score with ground truth")
    print("=" * 60)

    all_bn = []
    all_gt = {cls: [] for cls in range(NUM_CLASSES)}

    for rnum in ROUNDS:
        initial_states, gts = load_round_data(rnum)
        for seed in range(5):
            grid = initial_states[seed]["grid"]
            h = len(grid)
            w = len(grid[0])
            bn_grid = compute_bottleneck_grid(grid)

            gt = gts[seed]
            for y in range(h):
                for x in range(w):
                    code = grid[y][x]
                    if code in {10, 5}:
                        continue  # skip static
                    all_bn.append(bn_grid[y, x])
                    for cls in range(NUM_CLASSES):
                        all_gt[cls].append(gt[y, x, cls])

    all_bn = np.array(all_bn)
    print(f"\nTotal dynamic cells: {len(all_bn)}")
    print(f"Bottleneck score stats: mean={all_bn.mean():.3f}, std={all_bn.std():.3f}, "
          f"min={all_bn.min():.3f}, max={all_bn.max():.3f}")

    # Bin analysis
    bins = [0, 0.25, 0.5, 0.75, 0.9, 1.0]
    print(f"\nDistribution across bins:")
    for i in range(len(bins) - 1):
        mask = (all_bn >= bins[i]) & (all_bn < bins[i + 1])
        if i == len(bins) - 2:  # last bin is inclusive
            mask = (all_bn >= bins[i]) & (all_bn <= bins[i + 1])
        count = mask.sum()
        pct = 100 * count / len(all_bn)
        print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}]: {count:>6d} ({pct:>5.1f}%)")

    from dtos import CLASS_NAMES
    print(f"\nPearson correlation (bottleneck_score vs GT probability):")
    for cls in range(NUM_CLASSES):
        gt_arr = np.array(all_gt[cls])
        corr = np.corrcoef(all_bn, gt_arr)[0, 1]
        print(f"  {CLASS_NAMES[cls]:<12}: r = {corr:+.4f}")

    # Per-bin GT means (more informative than raw correlation)
    print(f"\nMean GT probability by bottleneck bin:")
    header = f"{'Bin':<16}"
    for cls in range(NUM_CLASSES):
        header += f"{CLASS_NAMES[cls]:>10}"
    print(header)

    for i in range(len(bins) - 1):
        mask = (all_bn >= bins[i]) & (all_bn < bins[i + 1])
        if i == len(bins) - 2:
            mask = (all_bn >= bins[i]) & (all_bn <= bins[i + 1])
        if mask.sum() == 0:
            continue
        row = f"[{bins[i]:.2f},{bins[i+1]:.2f}] "
        row += f"n={mask.sum():<5d} "
        for cls in range(NUM_CLASSES):
            gt_arr = np.array(all_gt[cls])
            row += f"{gt_arr[mask].mean():>10.4f}"
        print(row)

    # Per-terrain-type correlation
    print(f"\nCorrelation by terrain type:")
    for rnum in ROUNDS:
        break  # just use R1 for quick check
    initial_states, gts = load_round_data(1)
    for terrain_name, terrain_codes in [("Plains", {11, 0}), ("Forest", {4}), ("Settlement", {1, 2})]:
        bn_vals = []
        gt_settl = []
        for seed in range(5):
            grid = initial_states[seed]["grid"]
            bn_grid = compute_bottleneck_grid(grid)
            gt = gts[seed]
            h, w = len(grid), len(grid[0])
            for y in range(h):
                for x in range(w):
                    if grid[y][x] in terrain_codes:
                        bn_vals.append(bn_grid[y, x])
                        gt_settl.append(gt[y, x, 1])  # settlement prob
        bn_vals = np.array(bn_vals)
        gt_settl = np.array(gt_settl)
        if len(bn_vals) > 0:
            corr = np.corrcoef(bn_vals, gt_settl)[0, 1]
            print(f"  {terrain_name:<12}: r(bn, P(settl)) = {corr:+.4f}  (n={len(bn_vals)})")

    return all_bn


# ──────────────────────────────────────────────────────────────
# Step 2: LORO cross-validation
# ──────────────────────────────────────────────────────────────

def train_gbt_models(train_rounds, use_bottleneck=False):
    """Train terrain-specific XGBoost on given rounds."""
    X_data = {"plains": [], "forest": [], "settl": []}
    Y_data = {"plains": [], "forest": [], "settl": []}

    for rnum in train_rounds:
        initial_states, gts = load_round_data(rnum)
        for seed in range(5):
            grid = initial_states[seed]["grid"]
            settlements = initial_states[seed]["settlements"]
            gt = gts[seed]

            if use_bottleneck:
                feats, coords = extract_features_with_bottleneck(grid, settlements)
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

    models = {}
    for terrain_type in ["plains", "forest", "settl"]:
        X = np.array(X_data[terrain_type])
        Y = np.array(Y_data[terrain_type])
        terrain_models = []
        for cls in range(6):
            m = xgb.XGBRegressor(
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
            m.fit(X, Y[:, cls])
            terrain_models.append(m)
        models[terrain_type] = terrain_models

    return models


def gbt_predict_with_models(models_dict, initial_grid, settlements, use_bottleneck=False):
    """Generate GBT predictions using provided models."""
    h = len(initial_grid)
    w = len(initial_grid[0]) if h > 0 else 0

    if use_bottleneck:
        features, coords = extract_features_with_bottleneck(initial_grid, settlements)
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


def evaluate_round_with_blend(test_round_num, gbt_models, blend_weight, use_bottleneck=False):
    """Build prediction for a round and score it."""
    round_id = ROUNDS[test_round_num]
    initial_states, gts = load_round_data(test_round_num)
    all_observations = load_observations(round_id)

    scores = []
    for seed in range(5):
        grid = initial_states[seed]["grid"]
        settlements = initial_states[seed]["settlements"]
        gt = gts[seed]

        # Layer 1: Static prediction
        tensor = build_static_prediction(grid)

        # Layer 3: Fill unobserved dynamic
        tensor = fill_unobserved_dynamic(tensor, grid, settlements, [], seed)

        # Layer 6: GBT blend
        gbt_pred = gbt_predict_with_models(gbt_models, grid, settlements, use_bottleneck)
        if gbt_pred is not None:
            h, w, _ = tensor.shape
            for y in range(h):
                for x in range(w):
                    code = grid[y][x]
                    if code in {10, 5}:
                        continue
                    tensor[y, x] = (1 - blend_weight) * tensor[y, x] + blend_weight * gbt_pred[y, x]

        # Layer 7: Global observation ratio correction
        if all_observations:
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
                    adj = 1.0 + L7_STRENGTHS * (ratio - 1.0)
                    for y in range(h):
                        for x in range(w):
                            if grid[y][x] in {10, 5}:
                                continue
                            tensor[y, x] *= adj
                            tensor[y, x] = np.maximum(tensor[y, x], PROB_FLOOR)
                            tensor[y, x] /= tensor[y, x].sum()

        tensor = normalize_prediction(tensor)
        score = compute_score(tensor, gt)
        scores.append(score)

    return np.mean(scores), scores


def run_loro(use_bottleneck=False, label=""):
    """Run leave-one-round-out CV."""
    test_rounds = [1, 2, 4, 5]
    blend_weight = 0.30  # current production value

    print(f"\n{'='*60}")
    print(f"LORO CV: {label}")
    print(f"{'='*60}")

    header = f"{'Held-out':<10}"
    for r in test_rounds:
        header += f"{'R' + str(r):>8}"
    header += f"{'AVG':>8}"
    print(header)
    print("-" * 50)

    round_scores = []
    all_seed_scores = {}
    for held_out in test_rounds:
        train_on = [r for r in test_rounds if r != held_out]
        gbt_models = train_gbt_models(train_on, use_bottleneck=use_bottleneck)
        avg_score, seed_scores = evaluate_round_with_blend(
            held_out, gbt_models, blend_weight, use_bottleneck=use_bottleneck
        )
        round_scores.append(avg_score)
        all_seed_scores[held_out] = seed_scores
        seeds_str = " ".join(f"{s:.2f}" for s in seed_scores)
        print(f"  R{held_out}: avg={avg_score:.4f}  seeds=[{seeds_str}]")

    overall_avg = np.mean(round_scores)
    print(f"\n  LORO avg: {overall_avg:.4f}")
    return overall_avg, round_scores, all_seed_scores


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    # Step 1: Correlation analysis
    analyze_correlation()

    # Step 2: LORO comparison
    print("\n\n")
    baseline_avg, baseline_rounds, _ = run_loro(use_bottleneck=False, label="Baseline (no bottleneck)")
    bottleneck_avg, bottleneck_rounds, _ = run_loro(use_bottleneck=True, label="With bottleneck_score")

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline LORO avg:    {baseline_avg:.4f}")
    print(f"  Bottleneck LORO avg:  {bottleneck_avg:.4f}")
    diff = bottleneck_avg - baseline_avg
    print(f"  Delta:                {diff:+.4f}")
    if diff > 0:
        print(f"  Result: IMPROVEMENT (+{diff:.4f})")
    else:
        print(f"  Result: NO IMPROVEMENT ({diff:+.4f})")
