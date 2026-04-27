"""Experiment: food_availability feature.

food_availability = for each dynamic cell, find all settlements within Manhattan
distance 3. For each such settlement, count forest cells within radius 2 of that
settlement. Sum all those counts.

Hypothesis: cells near well-fed settlement clusters are more likely to be colonized.

Steps:
  1. Compute correlation with GT P(settlement) across all 5 rounds
  2. If correlation > 0.05, run LORO CV (baseline avg=89.68)
  3. Report numbers
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
)
from evaluate import compute_score
from utils import normalize_prediction, load_observations
from dtos import TERRAIN_TO_CLASS, NUM_CLASSES, PROB_FLOOR

# Round info — same as loro_blend_cv.py
ROUNDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
}

L7_STRENGTHS = np.array([1.38, 0.90, 0.44, 0.61, 1.16, 0.0])
GBT_BLEND_WEIGHT = 0.30

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_round_data(round_num):
    with open(os.path.join(DATA_DIR, f"round{round_num}_initial.json")) as f:
        info = json.load(f)
    initial_states = info["initial_states"]
    gts = []
    for seed in range(5):
        gt_path = os.path.join(DATA_DIR, f"gt_r{round_num}_seed{seed}.npy")
        gts.append(np.load(gt_path))
    return initial_states, gts


def compute_food_availability(initial_grid, settlements_list):
    """For each cell, sum forests-within-radius-2 of all settlements within dist 3.

    Args:
        initial_grid: H x W grid of terrain codes
        settlements_list: list of dicts with 'x', 'y' keys

    Returns:
        H x W array of food_availability values
    """
    h = len(initial_grid)
    w = len(initial_grid[0]) if h > 0 else 0
    grid_arr = np.array(initial_grid)

    # Get settlement positions
    settl_pos = []
    for s in settlements_list:
        if isinstance(s, dict):
            settl_pos.append((s.get("x", 0), s.get("y", 0)))
        else:
            settl_pos.append((s.x, s.y))

    # Pre-compute: for each settlement, count forests within radius 2
    settl_forest_count = {}
    for sx, sy in settl_pos:
        count = 0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dy == 0 and dx == 0:
                    continue
                if abs(dy) + abs(dx) > 2:
                    continue
                ny, nx = sy + dy, sx + dx
                if 0 <= ny < h and 0 <= nx < w and initial_grid[ny][nx] == 4:
                    count += 1
        settl_forest_count[(sx, sy)] = count

    # For each cell, sum forest counts of settlements within distance 3
    food_avail = np.zeros((h, w), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            total = 0
            for sx, sy in settl_pos:
                if abs(y - sy) + abs(x - sx) <= 3:
                    total += settl_forest_count[(sx, sy)]
            food_avail[y, x] = total

    return food_avail


def extract_features_with_food(grid, settlements):
    """Extract base features + food_availability appended."""
    base_feats, coords = _extract_cell_features(grid, settlements)
    if len(base_feats) == 0:
        return base_feats, coords

    food = compute_food_availability(grid, settlements)

    # Append food_availability as extra feature for each dynamic cell
    food_col = np.array([food[y, x] for y, x in coords]).reshape(-1, 1)
    augmented = np.hstack([base_feats, food_col])
    return augmented, coords


# ──────────────────────────────────────────────────────────────
# Step 1: Correlation analysis
# ──────────────────────────────────────────────────────────────

def correlation_analysis():
    """Compute correlation of food_availability with GT P(settlement) across all rounds."""
    print("=" * 60)
    print("STEP 1: Correlation of food_availability with GT P(settlement)")
    print("=" * 60)

    all_food = []
    all_p_settl = []

    for rnum in sorted(ROUNDS.keys()):
        initial_states, gts = load_round_data(rnum)
        r_food = []
        r_p_settl = []

        for seed in range(5):
            grid = initial_states[seed]["grid"]
            settlements = initial_states[seed]["settlements"]
            gt = gts[seed]
            h, w = len(grid), len(grid[0])

            food = compute_food_availability(grid, settlements)

            for y in range(h):
                for x in range(w):
                    code = grid[y][x]
                    if code in {10, 5}:  # skip static
                        continue
                    r_food.append(food[y, x])
                    r_p_settl.append(gt[y, x][1])  # class 1 = settlement

        corr = np.corrcoef(r_food, r_p_settl)[0, 1]
        print(f"  Round {rnum}: n={len(r_food):,} cells, corr={corr:.4f}")

        all_food.extend(r_food)
        all_p_settl.extend(r_p_settl)

    overall_corr = np.corrcoef(all_food, all_p_settl)[0, 1]
    print(f"\n  Overall: n={len(all_food):,} cells, corr={overall_corr:.4f}")

    # Also check correlation with other classes
    print("\n  Correlation with all GT classes (pooled across rounds):")
    for rnum in sorted(ROUNDS.keys()):
        initial_states, gts = load_round_data(rnum)
        class_corrs = [[] for _ in range(NUM_CLASSES)]
        food_vals = []

        for seed in range(5):
            grid = initial_states[seed]["grid"]
            settlements = initial_states[seed]["settlements"]
            gt = gts[seed]
            h, w = len(grid), len(grid[0])
            food = compute_food_availability(grid, settlements)

            for y in range(h):
                for x in range(w):
                    code = grid[y][x]
                    if code in {10, 5}:
                        continue
                    food_vals.append(food[y, x])
                    for cls in range(NUM_CLASSES):
                        class_corrs[cls].append(gt[y, x][cls])

        # Skip printing per-round class correlations — just do overall
        break  # we only need one example

    # Overall class correlations
    all_class_vals = [[] for _ in range(NUM_CLASSES)]
    all_food_vals = []
    CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

    for rnum in sorted(ROUNDS.keys()):
        initial_states, gts = load_round_data(rnum)
        for seed in range(5):
            grid = initial_states[seed]["grid"]
            settlements = initial_states[seed]["settlements"]
            gt = gts[seed]
            h, w = len(grid), len(grid[0])
            food = compute_food_availability(grid, settlements)

            for y in range(h):
                for x in range(w):
                    code = grid[y][x]
                    if code in {10, 5}:
                        continue
                    all_food_vals.append(food[y, x])
                    for cls in range(NUM_CLASSES):
                        all_class_vals[cls].append(gt[y, x][cls])

    for cls in range(NUM_CLASSES):
        c = np.corrcoef(all_food_vals, all_class_vals[cls])[0, 1]
        print(f"    {CLASS_NAMES[cls]:>12}: {c:+.4f}")

    return overall_corr


# ──────────────────────────────────────────────────────────────
# Step 2: LORO cross-validation
# ──────────────────────────────────────────────────────────────

def train_gbt_models(train_rounds, use_food=False):
    """Train terrain-specific XGBoost on given rounds."""
    X_data = {"plains": [], "forest": [], "settl": []}
    Y_data = {"plains": [], "forest": [], "settl": []}

    for rnum in train_rounds:
        initial_states, gts = load_round_data(rnum)
        for seed in range(5):
            grid = initial_states[seed]["grid"]
            settlements = initial_states[seed]["settlements"]
            gt = gts[seed]

            if use_food:
                feats, coords = extract_features_with_food(grid, settlements)
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


def gbt_predict_with_models(models_dict, initial_grid, settlements, use_food=False):
    """Generate GBT predictions using provided models."""
    h = len(initial_grid)
    w = len(initial_grid[0]) if h > 0 else 0

    if use_food:
        features, coords = extract_features_with_food(initial_grid, settlements)
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


def evaluate_round_with_blend(test_round_num, gbt_models, blend_weight, use_food=False):
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
        gbt_pred = gbt_predict_with_models(gbt_models, grid, settlements, use_food=use_food)
        if gbt_pred is not None:
            h, w, _ = tensor.shape
            for y in range(h):
                for x in range(w):
                    code = grid[y][x]
                    if code in {10, 5}:
                        continue
                    tensor[y, x] = (1 - blend_weight) * tensor[y, x] + blend_weight * gbt_pred[y, x]

        # Layer 7: Global observation ratio correction
        all_seed_obs = all_observations
        if all_seed_obs:
            obs_cls = np.zeros(NUM_CLASSES)
            obs_total = 0
            for obs in all_seed_obs:
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


def run_loro(use_food=False, label=""):
    """Run LORO cross-validation."""
    test_rounds = [1, 2, 4, 5]
    blend = GBT_BLEND_WEIGHT

    print(f"\n{'=' * 60}")
    print(f"LORO CV: {label}")
    print(f"{'=' * 60}")

    # Header
    print(f"{'Blend':<8}", end="")
    for r in test_rounds:
        print(f"{'R' + str(r):>8}", end="")
    print(f"{'AVG':>8}")
    print("-" * 48)

    round_scores = []
    round_details = {}
    for held_out in test_rounds:
        train_on = [r for r in test_rounds if r != held_out]
        gbt_models = train_gbt_models(train_on, use_food=use_food)
        avg, seeds = evaluate_round_with_blend(held_out, gbt_models, blend, use_food=use_food)
        round_scores.append(avg)
        round_details[held_out] = seeds
        seeds_str = " ".join(f"{s:.2f}" for s in seeds)
        print(f"  R{held_out}: avg={avg:.2f}  seeds=[{seeds_str}]")

    overall = np.mean(round_scores)
    print(f"\n{blend:<8.2f}", end="")
    for s in round_scores:
        print(f"{s:>8.2f}", end="")
    print(f"{overall:>8.2f}")

    return overall, round_scores, round_details


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    t0 = time.time()

    # Step 1: Correlation
    corr = correlation_analysis()

    # Step 2: LORO if correlation > 0.05
    if abs(corr) > 0.05:
        print(f"\nCorrelation {corr:.4f} > 0.05 threshold — running LORO CV")

        # Baseline (no food feature)
        baseline_avg, baseline_rounds, _ = run_loro(use_food=False, label="BASELINE (no food_availability)")

        # With food_availability
        food_avg, food_rounds, _ = run_loro(use_food=True, label="WITH food_availability")

        # Summary
        print(f"\n{'=' * 60}")
        print("COMPARISON")
        print(f"{'=' * 60}")
        print(f"  Baseline LORO avg:          {baseline_avg:.2f}")
        print(f"  With food_availability avg: {food_avg:.2f}")
        print(f"  Delta:                      {food_avg - baseline_avg:+.2f}")
        print(f"\n  Per-round deltas:")
        for i, rnum in enumerate([1, 2, 4, 5]):
            delta = food_rounds[i] - baseline_rounds[i]
            print(f"    R{rnum}: {baseline_rounds[i]:.2f} -> {food_rounds[i]:.2f} ({delta:+.2f})")
    else:
        print(f"\nCorrelation {corr:.4f} <= 0.05 threshold — skipping LORO CV")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
