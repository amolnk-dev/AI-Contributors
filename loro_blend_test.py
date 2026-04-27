"""LORO cross-validation for GBT blend weights.

Leave-One-Round-Out: for each test round, train XGBoost on the other 3 rounds,
build heuristic + GBT blended predictions, apply L7 correction, and score.

Usage:
    cd tasks/astar-island
    source ../../.venv/bin/activate
    ASTAR_TOKEN=... python loro_blend_test.py
"""

import json
import os
import sys
import pickle

import numpy as np
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))

from model import (
    _extract_cell_features,
    build_static_prediction,
    fill_unobserved_dynamic,
)
from evaluate import compute_score
from utils import normalize_prediction, load_observations
from dtos import TERRAIN_TO_CLASS, NUM_CLASSES, PROB_FLOOR

# ── Configuration ──────────────────────────────────────────────
ROUND_IDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    3: "f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
}

# Only use these rounds (skip r3)
ACTIVE_ROUNDS = [1, 2, 4, 5]

BLEND_WEIGHTS = [0.50, 0.60]

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


def load_round_data(round_num: int):
    """Load initial states and GT for a round."""
    with open(os.path.join(DATA_DIR, f"round{round_num}_initial.json")) as f:
        info = json.load(f)

    initial_states = info["initial_states"]
    seeds_count = info["seeds_count"]

    gt_list = []
    for seed in range(seeds_count):
        gt_path = os.path.join(DATA_DIR, f"gt_r{round_num}_seed{seed}.npy")
        gt = np.load(gt_path)
        gt_list.append(gt)

    return initial_states, gt_list, seeds_count


def extract_training_data(round_num: int):
    """Extract features and targets from one round's GT."""
    initial_states, gt_list, seeds_count = load_round_data(round_num)

    X_data = {"plains": [], "forest": [], "settl": []}
    Y_data = {"plains": [], "forest": [], "settl": []}

    for seed in range(seeds_count):
        state = initial_states[seed]
        grid = state["grid"]
        settlements = state["settlements"]
        gt = gt_list[seed]

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


def gbt_predict_with_models(initial_grid, settlements, models_dict):
    """Generate GBT predictions using given models (not the global ones)."""
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


def build_prediction_with_blend(initial_grid, settlements, gbt_models, blend_weight, all_observations):
    """Build full prediction: heuristic -> GBT blend -> L7 correction."""
    # Layer 1: static prediction
    tensor = build_static_prediction(initial_grid)

    # Layer 3: fill unobserved dynamic (pass empty obs to fill everything)
    tensor = fill_unobserved_dynamic(tensor, initial_grid, settlements, [], 0)

    # Layer 6: GBT blend (uniform blend weight)
    gbt_pred = gbt_predict_with_models(initial_grid, settlements, gbt_models)
    if gbt_pred is not None:
        h, w, _ = tensor.shape
        for y in range(h):
            for x in range(w):
                code = initial_grid[y][x]
                if code in {10, 5}:
                    continue
                tensor[y, x] = (1 - blend_weight) * tensor[y, x] + blend_weight * gbt_pred[y, x]

    # Layer 7: L7 correction
    tensor = apply_l7_correction(tensor, initial_grid, all_observations)

    # Final normalization
    tensor = normalize_prediction(tensor)
    return tensor


def main():
    print("LORO Cross-Validation: GBT Blend Weight Test")
    print("=" * 60)

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
        print(f"  Loaded R{rn}: {seeds_count} seeds, {len(obs)} obs")

    print()

    # Results table
    results = {}

    for blend in BLEND_WEIGHTS:
        print(f"\n--- Blend = {blend:.2f} ---")
        round_scores = {}

        for test_round in ACTIVE_ROUNDS:
            # Training rounds = all active except test round
            train_rounds = [r for r in ACTIVE_ROUNDS if r != test_round]

            # Collect training data from train rounds
            X_all = {"plains": [], "forest": [], "settl": []}
            Y_all = {"plains": [], "forest": [], "settl": []}

            for tr in train_rounds:
                X_r, Y_r = extract_training_data(tr)
                for ttype in ["plains", "forest", "settl"]:
                    X_all[ttype].extend(X_r[ttype])
                    Y_all[ttype].extend(Y_r[ttype])

            n_train = sum(len(X_all[t]) for t in X_all)
            print(f"  Test=R{test_round}, Train={train_rounds} ({n_train} samples) ... ", end="", flush=True)

            # Train GBT models
            gbt_models = train_gbt_models(X_all, Y_all)

            # Evaluate on test round
            test_data = round_data[test_round]
            seed_scores = []

            for seed in range(test_data["seeds_count"]):
                state = test_data["initial_states"][seed]
                grid = state["grid"]
                settlements = state["settlements"]
                gt = test_data["gt_list"][seed]
                obs = test_data["observations"]

                # Filter obs to this seed only
                seed_obs = [o for o in obs if o.get("seed_index") == seed]

                pred = build_prediction_with_blend(
                    grid, settlements, gbt_models, blend, seed_obs,
                )

                score = compute_score(pred, gt)
                seed_scores.append(score)

            avg = np.mean(seed_scores)
            round_scores[test_round] = avg
            print(f"score={avg:.2f}  (seeds: {', '.join(f'{s:.2f}' for s in seed_scores)})")

        overall_avg = np.mean(list(round_scores.values()))
        round_scores["AVG"] = overall_avg
        results[blend] = round_scores

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    header = f"{'blend':>6} | {'R1':>7} | {'R2':>7} | {'R4':>7} | {'R5':>7} | {'AVG':>7}"
    print(header)
    print("-" * len(header))
    for blend in BLEND_WEIGHTS:
        r = results[blend]
        print(f"{blend:>6.2f} | {r[1]:>7.2f} | {r[2]:>7.2f} | {r[4]:>7.2f} | {r[5]:>7.2f} | {r['AVG']:>7.2f}")


if __name__ == "__main__":
    main()
