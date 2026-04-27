"""Experiment: terrain_heterogeneity — Shannon entropy of terrain types in 5x5 window.

Hypothesis: Areas with mixed terrain (forest+plains+settlements) behave differently
from uniform areas. Expansion patterns depend on local terrain mix.

Formula: For each cell, count terrain types in 5x5 window, compute
  Shannon entropy = -sum(p * log(p))  where p = fraction of each terrain type.

Steps:
1. Compute feature and correlation with GT across all 5 rounds
2. Run LORO if promising (baseline avg=89.68)
3. Report numbers
"""

import json
import os
import sys
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

# ── Round configuration ──────────────────────────────────────

ROUND_IDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
}

ACTIVE_ROUNDS = [1, 2, 4, 5]

L7_STRENGTHS = np.array([1.38, 0.90, 0.44, 0.61, 1.16, 0.0])

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

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


# ── Terrain heterogeneity computation ────────────────────────

def compute_terrain_heterogeneity(initial_grid: list[list[int]], window: int = 5) -> np.ndarray:
    """Compute Shannon entropy of terrain types in a window around each cell.

    For each cell, count terrain type fractions in a (window x window) neighborhood,
    then compute H = -sum(p * log(p)).

    Returns H x W array of entropy values.
    """
    grid_arr = np.array(initial_grid)
    h, w = grid_arr.shape
    half = window // 2
    entropy_map = np.zeros((h, w), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            # Collect terrain codes in window
            y_lo = max(0, y - half)
            y_hi = min(h, y + half + 1)
            x_lo = max(0, x - half)
            x_hi = min(w, x + half + 1)

            patch = grid_arr[y_lo:y_hi, x_lo:x_hi].ravel()
            # Count unique terrain types
            unique, counts = np.unique(patch, return_counts=True)
            probs = counts / counts.sum()
            # Shannon entropy
            ent = -np.sum(probs * np.log(probs + 1e-12))
            entropy_map[y, x] = ent

    return entropy_map


# ── Feature extraction with terrain_heterogeneity ────────────

def extract_features_with_heterogeneity(initial_grid, settlements):
    """Extract base cell features + terrain_heterogeneity appended."""
    base_feats, coords = _extract_cell_features(initial_grid, settlements)
    if len(base_feats) == 0:
        return base_feats, coords

    het_map = compute_terrain_heterogeneity(initial_grid)
    het_values = np.array([het_map[y, x] for y, x in coords]).reshape(-1, 1)
    augmented = np.hstack([base_feats, het_values])
    return augmented, coords


# ── Data loading ─────────────────────────────────────────────

def load_round_data(round_num):
    with open(os.path.join(DATA_DIR, f"round{round_num}_initial.json")) as f:
        info = json.load(f)
    initial_states = info["initial_states"]
    seeds_count = info["seeds_count"]
    gt_list = []
    for seed in range(seeds_count):
        gt = np.load(os.path.join(DATA_DIR, f"gt_r{round_num}_seed{seed}.npy"))
        gt_list.append(gt)
    return initial_states, gt_list, seeds_count


# ── Step 1: Correlation analysis ─────────────────────────────

def analyze_correlation():
    """Compute correlation between terrain_heterogeneity and GT outcomes across all rounds."""
    print("=" * 60)
    print("STEP 1: Correlation of terrain_heterogeneity with GT")
    print("=" * 60)

    all_het = []
    all_gt_settlement = []
    all_gt_forest = []
    all_gt_ruin = []
    all_gt_entropy = []
    all_terrain_codes = []

    for rnum in ACTIVE_ROUNDS:
        initial_states, gt_list, seeds_count = load_round_data(rnum)
        for seed in range(seeds_count):
            grid = initial_states[seed]["grid"]
            gt = gt_list[seed]
            h, w = len(grid), len(grid[0])
            het_map = compute_terrain_heterogeneity(grid)

            for y in range(h):
                for x in range(w):
                    code = grid[y][x]
                    if code in {10, 5}:  # skip static
                        continue
                    all_het.append(het_map[y, x])
                    all_gt_settlement.append(gt[y, x, 1])  # P(settlement)
                    all_gt_forest.append(gt[y, x, 4])       # P(forest)
                    all_gt_ruin.append(gt[y, x, 3])          # P(ruin)
                    # GT entropy
                    p = gt[y, x]
                    p_safe = np.maximum(p, 1e-12)
                    ent = -np.sum(p_safe * np.log(p_safe))
                    all_gt_entropy.append(ent)
                    all_terrain_codes.append(code)

    all_het = np.array(all_het)
    all_gt_settlement = np.array(all_gt_settlement)
    all_gt_forest = np.array(all_gt_forest)
    all_gt_ruin = np.array(all_gt_ruin)
    all_gt_entropy = np.array(all_gt_entropy)
    all_terrain_codes = np.array(all_terrain_codes)

    # Overall correlations
    print(f"\nOverall (n={len(all_het)} dynamic cells across {len(ACTIVE_ROUNDS)} rounds):")
    print(f"  het range: [{all_het.min():.3f}, {all_het.max():.3f}], mean={all_het.mean():.3f}")
    print(f"  corr(het, P(settlement)):  {np.corrcoef(all_het, all_gt_settlement)[0,1]:+.4f}")
    print(f"  corr(het, P(forest)):      {np.corrcoef(all_het, all_gt_forest)[0,1]:+.4f}")
    print(f"  corr(het, P(ruin)):        {np.corrcoef(all_het, all_gt_ruin)[0,1]:+.4f}")
    print(f"  corr(het, GT entropy):     {np.corrcoef(all_het, all_gt_entropy)[0,1]:+.4f}")

    # Per-terrain correlations
    for tcode, tname in [(11, "Plains"), (4, "Forest"), (1, "Settlement"), (2, "Port")]:
        mask = all_terrain_codes == tcode
        n = mask.sum()
        if n < 10:
            continue
        print(f"\n  {tname} (code={tcode}, n={n}):")
        print(f"    corr(het, P(settlement)):  {np.corrcoef(all_het[mask], all_gt_settlement[mask])[0,1]:+.4f}")
        print(f"    corr(het, P(forest)):      {np.corrcoef(all_het[mask], all_gt_forest[mask])[0,1]:+.4f}")
        print(f"    corr(het, P(ruin)):        {np.corrcoef(all_het[mask], all_gt_ruin[mask])[0,1]:+.4f}")
        print(f"    corr(het, GT entropy):     {np.corrcoef(all_het[mask], all_gt_entropy[mask])[0,1]:+.4f}")

    # Binned analysis: does het predict outcomes?
    print(f"\n  Binned analysis (quintiles of terrain_heterogeneity):")
    pcts = [0, 20, 40, 60, 80, 100]
    thresholds = np.percentile(all_het, pcts)
    print(f"    {'Bin':<15} {'n':>6} {'P(settl)':>10} {'P(forest)':>10} {'P(ruin)':>10} {'GT_ent':>10}")
    for i in range(len(pcts) - 1):
        lo, hi = thresholds[i], thresholds[i+1]
        if i < len(pcts) - 2:
            mask = (all_het >= lo) & (all_het < hi)
        else:
            mask = (all_het >= lo) & (all_het <= hi)
        n = mask.sum()
        if n == 0:
            continue
        label = f"[{lo:.2f}, {hi:.2f})"
        print(f"    {label:<15} {n:>6} {all_gt_settlement[mask].mean():>10.4f} "
              f"{all_gt_forest[mask].mean():>10.4f} {all_gt_ruin[mask].mean():>10.4f} "
              f"{all_gt_entropy[mask].mean():>10.4f}")

    return np.corrcoef(all_het, all_gt_settlement)[0, 1]


# ── Step 2: LORO evaluation ─────────────────────────────────

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


def gbt_predict_with_models(initial_grid, settlements, models_dict, use_het=False):
    """Generate GBT predictions using given models."""
    h = len(initial_grid)
    w = len(initial_grid[0]) if h > 0 else 0

    if use_het:
        features, coords = extract_features_with_heterogeneity(initial_grid, settlements)
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


def loro_evaluate(use_het=False, blend_weight=0.30):
    """Leave-One-Round-Out cross-validation.

    use_het: whether to include terrain_heterogeneity as an extra XGBoost feature.
    """
    label = "WITH terrain_heterogeneity" if use_het else "BASELINE (no het)"
    print(f"\n--- LORO: {label}, blend={blend_weight} ---")

    round_data = {}
    for rnum in ACTIVE_ROUNDS:
        initial_states, gt_list, seeds_count = load_round_data(rnum)
        obs = load_observations(ROUND_IDS[rnum])
        round_data[rnum] = {
            "initial_states": initial_states,
            "gt_list": gt_list,
            "seeds_count": seeds_count,
            "observations": obs,
        }

    round_scores = {}
    for test_round in ACTIVE_ROUNDS:
        train_rounds = [r for r in ACTIVE_ROUNDS if r != test_round]

        # Collect training data
        X_data = {"plains": [], "forest": [], "settl": []}
        Y_data = {"plains": [], "forest": [], "settl": []}

        for rn in train_rounds:
            rd = round_data[rn]
            for seed in range(rd["seeds_count"]):
                grid = rd["initial_states"][seed]["grid"]
                settlements = rd["initial_states"][seed]["settlements"]
                gt = rd["gt_list"][seed]

                if use_het:
                    feats, coords = extract_features_with_heterogeneity(grid, settlements)
                else:
                    feats, coords = _extract_cell_features(grid, settlements)

                targets = np.array([gt[y, x] for y, x in coords])

                for i, (y, x) in enumerate(coords):
                    code = grid[y][x]
                    if code in {11, 0}:
                        ttype = "plains"
                    elif code == 4:
                        ttype = "forest"
                    elif code in {1, 2}:
                        ttype = "settl"
                    else:
                        ttype = "plains"
                    X_data[ttype].append(feats[i])
                    Y_data[ttype].append(targets[i])

        # Train
        gbt_models = train_gbt_models(X_data, Y_data)

        # Evaluate on test round
        test_data = round_data[test_round]
        seed_scores = []

        for seed in range(test_data["seeds_count"]):
            grid = test_data["initial_states"][seed]["grid"]
            settlements = test_data["initial_states"][seed]["settlements"]
            gt = test_data["gt_list"][seed]
            obs = test_data["observations"]
            seed_obs = [o for o in obs if o.get("seed_index") == seed]

            # Build prediction
            tensor = build_static_prediction(grid)
            tensor = fill_unobserved_dynamic(tensor, grid, settlements, [], 0)

            gbt_pred = gbt_predict_with_models(grid, settlements, gbt_models, use_het=use_het)
            if gbt_pred is not None:
                h, w, _ = tensor.shape
                for y in range(h):
                    for x in range(w):
                        code = grid[y][x]
                        if code in {10, 5}:
                            continue
                        tensor[y, x] = (1 - blend_weight) * tensor[y, x] + blend_weight * gbt_pred[y, x]

            # L7 correction
            tensor = apply_l7_correction(tensor, grid, seed_obs)
            tensor = normalize_prediction(tensor)

            score = compute_score(tensor, gt)
            seed_scores.append(score)

        avg = np.mean(seed_scores)
        round_scores[test_round] = avg
        seeds_str = ", ".join(f"{s:.2f}" for s in seed_scores)
        print(f"  Test=R{test_round}: avg={avg:.2f}  (seeds: {seeds_str})")

    overall = np.mean(list(round_scores.values()))
    print(f"  LORO avg: {overall:.4f}")
    return round_scores, overall


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    # Step 1: Correlation analysis
    corr = analyze_correlation()

    # Step 2: LORO (always run to get definitive numbers)
    print("\n\n" + "=" * 60)
    print("STEP 2: LORO Cross-Validation")
    print("=" * 60)

    baseline_scores, baseline_avg = loro_evaluate(use_het=False, blend_weight=0.30)
    het_scores, het_avg = loro_evaluate(use_het=True, blend_weight=0.30)

    # Summary
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Baseline LORO avg:              {baseline_avg:.4f}")
    print(f"  With terrain_heterogeneity avg: {het_avg:.4f}")
    print(f"  Delta:                          {het_avg - baseline_avg:+.4f}")
    print()
    print(f"  Per-round comparison:")
    for rn in ACTIVE_ROUNDS:
        b = baseline_scores[rn]
        h = het_scores[rn]
        print(f"    R{rn}: baseline={b:.2f}, het={h:.2f}, delta={h-b:+.2f}")

    if het_avg > baseline_avg:
        print(f"\n  RESULT: terrain_heterogeneity IMPROVES LORO by {het_avg - baseline_avg:+.4f}")
    else:
        print(f"\n  RESULT: terrain_heterogeneity DOES NOT IMPROVE LORO ({het_avg - baseline_avg:+.4f})")
