"""Experiment: Round-level observation features in XGBoost.

Tests whether appending observation-derived round-level statistics
(settlement rate, port rate, etc.) to the XGBoost feature vector
helps the model adapt to each round's hidden parameters.

Approach:
1. Baseline: current model (static XGBoost + Layer 7 post-hoc correction)
2. Approach A: XGBoost with obs features appended to every cell
3. Approach B: Linear scaling of settlement probability based on obs rate
4. Approach C: Interaction features (obs_rate * cell features)

Evaluation: Leave-one-round-out cross-validation on R1, R2, R4.
"""

import json
import os
import pickle
import sys
import time

import numpy as np
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model import (
    _extract_cell_features,
    build_static_prediction,
    fill_unobserved_dynamic,
    GBT_BLEND_WEIGHT,
)
from utils import normalize_prediction, load_observations
from evaluate import compute_score
from dtos import NUM_CLASSES, PROB_FLOOR, TERRAIN_TO_CLASS


# ──────────────────────────────────────────────────────────────
# Round definitions
# ──────────────────────────────────────────────────────────────

ROUNDS = {
    "R1": {"id": "71451d74-be9f-471f-aacd-a41f3b68a9cd", "num": 1},
    "R2": {"id": "76909e29-f664-4b2f-b16b-61b7507277e9", "num": 2},
    "R4": {"id": "8e839974-b13b-407b-a5e7-fc749d877195", "num": 4},
}


def load_round_data(rname):
    """Load initial states and GT for a round."""
    rinfo = ROUNDS[rname]
    rnum = rinfo["num"]
    rid = rinfo["id"]

    with open(f"data/round{rnum}_initial.json") as f:
        init = json.load(f)

    obs = load_observations(rid)
    all_grids = [s["grid"] for s in init["initial_states"]]
    settlements_list = [s["settlements"] for s in init["initial_states"]]

    gts = []
    for seed in range(5):
        gt = np.load(f"data/gt_r{rnum}_seed{seed}.npy")
        gts.append(gt)

    return {
        "grids": all_grids,
        "settlements": settlements_list,
        "gts": gts,
        "obs": obs,
        "name": rname,
        "num": rnum,
        "id": rid,
    }


def compute_obs_features(observations):
    """Compute round-level observation statistics.

    Returns a dict with observation-derived features.
    These are the same for every cell in the round.
    """
    obs_cls = np.zeros(NUM_CLASSES)
    obs_total = 0
    pop_values = []
    n_settlements_obs = 0

    for ob in observations:
        for row in ob.get("grid", []):
            for code in row:
                if code not in {10, 5}:  # skip static
                    obs_cls[TERRAIN_TO_CLASS.get(code, 0)] += 1
                    obs_total += 1
        for s in ob.get("settlements", []):
            if isinstance(s, dict):
                pop = s.get("population", 0)
                if pop:
                    pop_values.append(pop)
                n_settlements_obs += 1

    if obs_total > 0:
        obs_freq = obs_cls / obs_total
    else:
        obs_freq = np.ones(NUM_CLASSES) / NUM_CLASSES

    avg_pop = np.mean(pop_values) if pop_values else 0.0

    return {
        "obs_empty_rate": obs_freq[0],
        "obs_settl_rate": obs_freq[1],
        "obs_port_rate": obs_freq[2],
        "obs_ruin_rate": obs_freq[3],
        "obs_forest_rate": obs_freq[4],
        "obs_avg_population": avg_pop,
        "obs_n_settlements": n_settlements_obs / max(len(observations), 1),
        "obs_total_cells": obs_total,
    }


def extract_features_with_obs(grid, settlements, obs_feats):
    """Extract cell features + append round-level obs features."""
    base_feats, coords = _extract_cell_features(grid, settlements)
    if len(base_feats) == 0:
        return base_feats, coords

    n_cells = base_feats.shape[0]
    # Append obs features to every cell
    obs_vec = np.array([
        obs_feats["obs_settl_rate"],
        obs_feats["obs_port_rate"],
        obs_feats["obs_ruin_rate"],
        obs_feats["obs_forest_rate"],
        obs_feats["obs_avg_population"],
        obs_feats["obs_n_settlements"],
    ])
    obs_broadcast = np.tile(obs_vec, (n_cells, 1))
    augmented = np.hstack([base_feats, obs_broadcast])
    return augmented, coords


def extract_features_with_obs_interactions(grid, settlements, obs_feats):
    """Extract cell features + obs features + interaction terms."""
    base_feats, coords = _extract_cell_features(grid, settlements)
    if len(base_feats) == 0:
        return base_feats, coords

    n_cells = base_feats.shape[0]
    obs_settl = obs_feats["obs_settl_rate"]
    obs_forest = obs_feats["obs_forest_rate"]
    obs_ruin = obs_feats["obs_ruin_rate"]
    obs_port = obs_feats["obs_port_rate"]
    obs_pop = obs_feats["obs_avg_population"]
    obs_n_settl = obs_feats["obs_n_settlements"]

    # Base obs features
    obs_vec = np.array([obs_settl, obs_port, obs_ruin, obs_forest, obs_pop, obs_n_settl])
    obs_broadcast = np.tile(obs_vec, (n_cells, 1))

    # Interaction features: obs rates * key cell features
    # Feature indices from _extract_cell_features:
    #   1 = dist_to_nearest_settlement
    #   4 = adj_settl
    #   26 = bfs_dist
    #   19 = settlements_r3
    dist_col = base_feats[:, 1:2]         # dist to nearest settlement
    adj_settl_col = base_feats[:, 4:5]    # adjacent settlements
    bfs_col = base_feats[:, 26:27]        # BFS distance
    settl_r3_col = base_feats[:, 19:20]   # settlements in radius 3

    interactions = np.hstack([
        obs_settl * dist_col,       # obs_settl * dist
        obs_settl * adj_settl_col,  # obs_settl * adj_settl
        obs_settl * bfs_col,        # obs_settl * bfs_dist
        obs_settl * settl_r3_col,   # obs_settl * settl_r3
        obs_forest * dist_col,      # obs_forest * dist
        obs_ruin * dist_col,        # obs_ruin * dist
        obs_settl / (1 + dist_col), # obs_settl / (1 + dist)
    ])

    augmented = np.hstack([base_feats, obs_broadcast, interactions])
    return augmented, coords


def train_terrain_xgb(X_data, Y_data, n_estimators=200, max_depth=4):
    """Train terrain-specific XGBoost models."""
    models = {}
    for terrain_type in ["plains", "forest", "settl"]:
        X = np.array(X_data[terrain_type])
        Y = np.array(Y_data[terrain_type])
        if len(X) == 0:
            continue

        terrain_models = []
        for cls in range(NUM_CLASSES):
            m = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
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


def predict_with_models(models, features, coords, grid, h, w):
    """Generate H x W x 6 tensor from terrain-specific models."""
    tensor = np.zeros((h, w, NUM_CLASSES))
    # Static cells
    for y in range(h):
        for x in range(w):
            if grid[y][x] == 10:
                tensor[y, x] = [1, 0, 0, 0, 0, 0]
            elif grid[y][x] == 5:
                tensor[y, x] = [0, 0, 0, 0, 0, 1]

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

        models_t = models.get(ttype)
        if models_t is None:
            continue

        pred_v = np.zeros(NUM_CLASSES)
        for cls in range(NUM_CLASSES):
            pred_v[cls] = models_t[cls].predict(features[i:i + 1])[0]
        tensor[y, x] = np.maximum(pred_v, PROB_FLOOR)
        tensor[y, x] /= tensor[y, x].sum()

    return tensor


def build_prediction_with_gbt(grid, settlements, gbt_models, features, coords,
                               obs=None, apply_layer7=False, obs_feats=None):
    """Build full prediction using custom GBT models.

    Mimics build_prediction from model.py but uses provided GBT models.
    """
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0

    # Layer 1: Static prediction
    tensor = build_static_prediction(grid)

    # Layer 3: Fill with context priors
    tensor = fill_unobserved_dynamic(tensor, grid, settlements, [], 0)

    # Layer 6: GBT blend
    gbt_pred = predict_with_models(gbt_models, features, coords, grid, h, w)
    tensor = (1 - GBT_BLEND_WEIGHT) * tensor + GBT_BLEND_WEIGHT * gbt_pred

    # Layer 7: Post-hoc ratio correction (optional)
    if apply_layer7 and obs:
        obs_cls = np.zeros(NUM_CLASSES)
        obs_total = 0
        for ob in obs:
            for row in ob.get("grid", []):
                for code in row:
                    if code not in {10, 5}:
                        obs_cls[TERRAIN_TO_CLASS.get(code, 0)] += 1
                        obs_total += 1

        if obs_total > 100:
            obs_freq = obs_cls / obs_total
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
                cls_strength = np.array([0.2, 0.6, 0.2, 0.6, 0.4, 0.0])
                adj = 1.0 + cls_strength * (ratio - 1.0)
                for y in range(h):
                    for x in range(w):
                        if grid[y][x] in {10, 5}:
                            continue
                        tensor[y, x] *= adj
                        tensor[y, x] = np.maximum(tensor[y, x], PROB_FLOOR)
                        tensor[y, x] /= tensor[y, x].sum()

    tensor = normalize_prediction(tensor)
    return tensor


# ──────────────────────────────────────────────────────────────
# Leave-one-round-out evaluation
# ──────────────────────────────────────────────────────────────

def loo_evaluate(approach="baseline"):
    """Leave-one-round-out cross-validation.

    For each test round, train on the other two rounds and evaluate.
    approach: "baseline", "obs_features", "obs_interactions", "linear_scaling"
    """
    round_names = ["R1", "R2", "R4"]
    data = {rn: load_round_data(rn) for rn in round_names}

    results = {}

    for test_round in round_names:
        train_rounds = [rn for rn in round_names if rn != test_round]
        test_data = data[test_round]

        # Compute obs features for each round
        obs_features = {}
        for rn in round_names:
            obs_features[rn] = compute_obs_features(data[rn]["obs"])

        print(f"\n--- Test: {test_round}, Train: {train_rounds} ---")
        print(f"  Test obs_settl_rate: {obs_features[test_round]['obs_settl_rate']:.4f}")
        for tr in train_rounds:
            print(f"  Train {tr} obs_settl_rate: {obs_features[tr]['obs_settl_rate']:.4f}")

        # Collect training data
        X_data = {"plains": [], "forest": [], "settl": []}
        Y_data = {"plains": [], "forest": [], "settl": []}

        for rn in train_rounds:
            rd = data[rn]
            obs_f = obs_features[rn]

            for seed in range(5):
                grid = rd["grids"][seed]
                settlements = rd["settlements"][seed]
                gt = rd["gts"][seed]

                if approach in ("obs_features", "obs_features_plus_l7"):
                    feats, coords = extract_features_with_obs(grid, settlements, obs_f)
                elif approach == "obs_interactions":
                    feats, coords = extract_features_with_obs_interactions(grid, settlements, obs_f)
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

        # Train models
        t0 = time.time()
        models = train_terrain_xgb(X_data, Y_data)
        train_time = time.time() - t0
        for ttype in X_data:
            print(f"  {ttype}: {len(X_data[ttype])} training samples")
        print(f"  Training time: {train_time:.1f}s")

        # Evaluate on test round
        test_obs_f = obs_features[test_round]
        scores = []

        for seed in range(5):
            grid = test_data["grids"][seed]
            settlements = test_data["settlements"][seed]
            gt = test_data["gts"][seed]
            h, w = len(grid), len(grid[0])

            if approach in ("obs_features", "obs_features_plus_l7"):
                feats, coords = extract_features_with_obs(grid, settlements, test_obs_f)
            elif approach == "obs_interactions":
                feats, coords = extract_features_with_obs_interactions(grid, settlements, test_obs_f)
            else:
                feats, coords = _extract_cell_features(grid, settlements)

            if approach == "baseline":
                # Baseline: no obs features, WITH Layer 7 correction
                pred = build_prediction_with_gbt(
                    grid, settlements, models, feats, coords,
                    obs=test_data["obs"], apply_layer7=True,
                )
            elif approach == "baseline_no_l7":
                # Baseline without Layer 7
                pred = build_prediction_with_gbt(
                    grid, settlements, models, feats, coords,
                    obs=None, apply_layer7=False,
                )
            elif approach in ("obs_features", "obs_interactions"):
                # XGBoost has obs features — test both with and without L7
                pred = build_prediction_with_gbt(
                    grid, settlements, models, feats, coords,
                    obs=None, apply_layer7=False,
                )
            elif approach == "obs_features_plus_l7":
                # feats already computed with obs features in the loop above
                pred = build_prediction_with_gbt(
                    grid, settlements, models, feats, coords,
                    obs=test_data["obs"], apply_layer7=True,
                )
            elif approach == "linear_scaling":
                # Train without obs features, apply linear scaling after
                feats_base, coords_base = _extract_cell_features(grid, settlements)
                pred = build_prediction_with_gbt(
                    grid, settlements, models, feats_base, coords_base,
                    obs=None, apply_layer7=False,
                )
                # Apply linear scaling based on obs vs model settlement rate
                obs_settl = test_obs_f["obs_settl_rate"]
                model_avg = np.zeros(NUM_CLASSES)
                m_count = 0
                for y in range(h):
                    for x in range(w):
                        if grid[y][x] not in {10, 5}:
                            model_avg += pred[y, x]
                            m_count += 1
                if m_count > 0:
                    model_avg /= m_count
                    ratio = np.array([
                        test_obs_f["obs_empty_rate"],
                        test_obs_f["obs_settl_rate"],
                        test_obs_f["obs_port_rate"],
                        test_obs_f["obs_ruin_rate"],
                        test_obs_f["obs_forest_rate"],
                        0.0,
                    ]) / np.maximum(model_avg, 1e-6)
                    # Stronger correction for all classes
                    cls_strength = np.array([0.3, 0.7, 0.3, 0.7, 0.5, 0.0])
                    adj_vec = 1.0 + cls_strength * (ratio - 1.0)
                    for y in range(h):
                        for x in range(w):
                            if grid[y][x] in {10, 5}:
                                continue
                            pred[y, x] *= adj_vec
                            pred[y, x] = np.maximum(pred[y, x], PROB_FLOOR)
                            pred[y, x] /= pred[y, x].sum()
                pred = normalize_prediction(pred)

            s = compute_score(pred, gt)
            scores.append(s)

        avg = np.mean(scores)
        results[test_round] = {"avg": avg, "scores": scores}
        scores_str = " ".join(f"{s:.2f}" for s in scores)
        print(f"  Test {test_round}: avg={avg:.4f}  seeds=[{scores_str}]")

    # Summary
    overall = np.mean([r["avg"] for r in results.values()])
    print(f"\n{'='*60}")
    print(f"APPROACH: {approach}")
    print(f"Overall LOO avg: {overall:.4f}")
    for rn, r in results.items():
        print(f"  {rn}: {r['avg']:.4f}")
    print(f"{'='*60}")

    return results, overall


# ──────────────────────────────────────────────────────────────
# Also test the CURRENT model (uses the pre-saved gbt_models.pkl)
# ──────────────────────────────────────────────────────────────

def evaluate_current_model():
    """Evaluate the current model.py pipeline (with pre-saved GBT + L7)."""
    from model import build_prediction

    round_names = ["R1", "R2", "R4"]
    results = {}

    for rname in round_names:
        rinfo = ROUNDS[rname]
        rnum = rinfo["num"]
        rid = rinfo["id"]

        with open(f"data/round{rnum}_initial.json") as f:
            init = json.load(f)

        obs = load_observations(rid)
        all_grids = [s["grid"] for s in init["initial_states"]]

        scores = []
        for seed in range(5):
            gt = np.load(f"data/gt_r{rnum}_seed{seed}.npy")
            grid = all_grids[seed]
            settlements = init["initial_states"][seed]["settlements"]
            same_seed_obs = [o for o in obs if o["seed_index"] == seed]

            pred = build_prediction(
                initial_grid=grid,
                settlements=settlements,
                observations=same_seed_obs,
                seed_index=seed,
                all_initial_grids=all_grids,
                all_observations=same_seed_obs,
            )
            pred = normalize_prediction(pred)
            s = compute_score(pred, gt)
            scores.append(s)

        avg = np.mean(scores)
        results[rname] = {"avg": avg, "scores": scores}
        scores_str = " ".join(f"{s:.2f}" for s in scores)
        print(f"{rname}: avg={avg:.4f}  seeds=[{scores_str}]")

    overall = np.mean([r["avg"] for r in results.values()])
    print(f"\nCurrent model overall (R1+R2+R4): {overall:.4f}")
    return results, overall


def print_comparison(approaches_list):
    """Print comparison table."""
    print("\n\n")
    print("=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Approach':<35} {'R1':>8} {'R2':>8} {'R4':>8} {'Avg':>8}")
    print("-" * 70)
    for name, results, overall in approaches_list:
        r1 = results.get("R1", {}).get("avg", 0)
        r2 = results.get("R2", {}).get("avg", 0)
        r4 = results.get("R4", {}).get("avg", 0)
        print(f"{name:<35} {r1:>8.2f} {r2:>8.2f} {r4:>8.2f} {overall:>8.2f}")
    print("=" * 70)


def loo_evaluate_l7_sweep():
    """Test different L7 strengths with the baseline LOO setup."""
    round_names = ["R1", "R2", "R4"]
    data = {rn: load_round_data(rn) for rn in round_names}
    obs_features = {}
    for rn in round_names:
        obs_features[rn] = compute_obs_features(data[rn]["obs"])

    # Different L7 class strength configs to test
    configs = {
        "L7 weak":    np.array([0.1, 0.3, 0.1, 0.3, 0.2, 0.0]),
        "L7 current": np.array([0.2, 0.6, 0.2, 0.6, 0.4, 0.0]),
        "L7 strong":  np.array([0.3, 0.8, 0.3, 0.8, 0.6, 0.0]),
        "L7 full":    np.array([0.5, 1.0, 0.5, 1.0, 0.8, 0.0]),
    }

    all_results = {}

    for config_name, cls_strength in configs.items():
        results = {}
        for test_round in round_names:
            train_rounds = [rn for rn in round_names if rn != test_round]

            # Train
            X_data = {"plains": [], "forest": [], "settl": []}
            Y_data = {"plains": [], "forest": [], "settl": []}
            for rn in train_rounds:
                rd = data[rn]
                for seed in range(5):
                    feats, coords = _extract_cell_features(
                        rd["grids"][seed], rd["settlements"][seed])
                    gt = rd["gts"][seed]
                    targets = np.array([gt[y, x] for y, x in coords])
                    for i, (y, x) in enumerate(coords):
                        code = rd["grids"][seed][y][x]
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

            models = train_terrain_xgb(X_data, Y_data)

            # Eval with custom L7 strength
            test_data = data[test_round]
            scores = []
            for seed in range(5):
                grid = test_data["grids"][seed]
                settlements = test_data["settlements"][seed]
                gt = test_data["gts"][seed]
                h, w = len(grid), len(grid[0])
                feats, coords = _extract_cell_features(grid, settlements)

                # Build pred without L7
                tensor = build_static_prediction(grid)
                tensor = fill_unobserved_dynamic(tensor, grid, settlements, [], 0)
                gbt_pred = predict_with_models(models, feats, coords, grid, h, w)
                tensor = (1 - GBT_BLEND_WEIGHT) * tensor + GBT_BLEND_WEIGHT * gbt_pred

                # Custom L7
                obs = test_data["obs"]
                obs_cls = np.zeros(NUM_CLASSES)
                obs_total = 0
                for ob in obs:
                    for row in ob.get("grid", []):
                        for code_val in row:
                            if code_val not in {10, 5}:
                                obs_cls[TERRAIN_TO_CLASS.get(code_val, 0)] += 1
                                obs_total += 1
                if obs_total > 100:
                    obs_freq = obs_cls / obs_total
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
                        adj = 1.0 + cls_strength * (ratio - 1.0)
                        for y in range(h):
                            for x in range(w):
                                if grid[y][x] in {10, 5}:
                                    continue
                                tensor[y, x] *= adj
                                tensor[y, x] = np.maximum(tensor[y, x], PROB_FLOOR)
                                tensor[y, x] /= tensor[y, x].sum()

                tensor = normalize_prediction(tensor)
                scores.append(compute_score(tensor, gt))

            avg = np.mean(scores)
            results[test_round] = {"avg": avg, "scores": scores}

        overall = np.mean([r["avg"] for r in results.values()])
        all_results[config_name] = (results, overall)
        scores_str = " | ".join(
            f"{rn}={results[rn]['avg']:.2f}" for rn in round_names
        )
        print(f"  {config_name}: {scores_str} | avg={overall:.2f}")

    return all_results


def loo_evaluate_obs_interactions_plus_l7():
    """Best obs approach + L7: obs_interactions features AND Layer 7 correction."""
    round_names = ["R1", "R2", "R4"]
    data = {rn: load_round_data(rn) for rn in round_names}
    obs_features = {}
    for rn in round_names:
        obs_features[rn] = compute_obs_features(data[rn]["obs"])

    results = {}
    for test_round in round_names:
        train_rounds = [rn for rn in round_names if rn != test_round]

        X_data = {"plains": [], "forest": [], "settl": []}
        Y_data = {"plains": [], "forest": [], "settl": []}
        for rn in train_rounds:
            rd = data[rn]
            obs_f = obs_features[rn]
            for seed in range(5):
                feats, coords = extract_features_with_obs_interactions(
                    rd["grids"][seed], rd["settlements"][seed], obs_f)
                gt = rd["gts"][seed]
                targets = np.array([gt[y, x] for y, x in coords])
                for i, (y, x) in enumerate(coords):
                    code = rd["grids"][seed][y][x]
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

        models = train_terrain_xgb(X_data, Y_data)
        test_data = data[test_round]
        test_obs_f = obs_features[test_round]

        scores = []
        for seed in range(5):
            grid = test_data["grids"][seed]
            settlements = test_data["settlements"][seed]
            gt = test_data["gts"][seed]
            feats, coords = extract_features_with_obs_interactions(
                grid, settlements, test_obs_f)
            pred = build_prediction_with_gbt(
                grid, settlements, models, feats, coords,
                obs=test_data["obs"], apply_layer7=True,
            )
            scores.append(compute_score(pred, gt))

        avg = np.mean(scores)
        results[test_round] = {"avg": avg, "scores": scores}

    overall = np.mean([r["avg"] for r in results.values()])
    return results, overall


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    all_approaches = []

    print("=" * 60)
    print("CURRENT MODEL (pre-trained GBT + Layer 7)")
    print("=" * 60)
    current_results, current_overall = evaluate_current_model()
    all_approaches.append(("Current model (all 4 rounds)", current_results, current_overall))

    print("\n\n")
    print("=" * 60)
    print("LOO: BASELINE (retrained GBT on 2 rounds + Layer 7)")
    print("=" * 60)
    baseline_results, baseline_overall = loo_evaluate("baseline")
    all_approaches.append(("LOO: Baseline + L7", baseline_results, baseline_overall))

    print("\n\n")
    print("=" * 60)
    print("LOO: BASELINE WITHOUT LAYER 7")
    print("=" * 60)
    no_l7_results, no_l7_overall = loo_evaluate("baseline_no_l7")
    all_approaches.append(("LOO: Baseline (no L7)", no_l7_results, no_l7_overall))

    print("\n\n")
    print("=" * 60)
    print("LOO: OBS FEATURES (6 round-level features appended)")
    print("=" * 60)
    obs_results, obs_overall = loo_evaluate("obs_features")
    all_approaches.append(("LOO: Obs features", obs_results, obs_overall))

    print("\n\n")
    print("=" * 60)
    print("LOO: OBS INTERACTIONS (obs + interaction terms)")
    print("=" * 60)
    int_results, int_overall = loo_evaluate("obs_interactions")
    all_approaches.append(("LOO: Obs interactions", int_results, int_overall))

    print("\n\n")
    print("=" * 60)
    print("LOO: LINEAR SCALING (retrained GBT + obs-based scaling)")
    print("=" * 60)
    ls_results, ls_overall = loo_evaluate("linear_scaling")
    all_approaches.append(("LOO: Linear scaling", ls_results, ls_overall))

    print("\n\n")
    print("=" * 60)
    print("LOO: OBS FEATURES + LAYER 7")
    print("=" * 60)
    obs_l7_results, obs_l7_overall = loo_evaluate("obs_features_plus_l7")
    all_approaches.append(("LOO: Obs features + L7", obs_l7_results, obs_l7_overall))

    print("\n\n")
    print("=" * 60)
    print("LOO: OBS INTERACTIONS + LAYER 7")
    print("=" * 60)
    int_l7_results, int_l7_overall = loo_evaluate_obs_interactions_plus_l7()
    all_approaches.append(("LOO: Obs interactions + L7", int_l7_results, int_l7_overall))

    print("\n\n")
    print("=" * 60)
    print("L7 STRENGTH SWEEP (with LOO baseline)")
    print("=" * 60)
    l7_sweep = loo_evaluate_l7_sweep()
    for config_name, (results, overall) in l7_sweep.items():
        all_approaches.append((f"LOO: {config_name}", results, overall))

    print_comparison(all_approaches)
