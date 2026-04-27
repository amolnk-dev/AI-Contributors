"""LORO (Leave-One-Round-Out) cross-validation for GBT blend weights.

For each blend weight and each held-out round:
  1. Train terrain-specific XGBoost on the other 3 rounds' GT data
  2. Build heuristic prediction
  3. Blend: pred = (1-blend)*heuristic + blend*XGBoost
  4. Apply L7 observation ratio correction
  5. Score against held-out round's GT
"""

import json
import os
import sys
import pickle
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
from simulator import simulate_monte_carlo, fit_hidden_params
from evaluate import compute_score
from utils import normalize_prediction, load_observations
from dtos import TERRAIN_TO_CLASS, NUM_CLASSES, PROB_FLOOR

# Round info
ROUNDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    6: "ae78003a-4efe-425a-881a-d16a39bca0ad",
}

# L7 strengths
L7_STRENGTHS = np.array([1.20, 0.80, 0.44, 0.80, 1.30, 0.0])

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


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


def gbt_predict_with_models(models_dict, initial_grid, settlements):
    """Generate GBT predictions using provided models (not the global ones)."""
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


def evaluate_round_with_blend(test_round_num, gbt_models, blend_weight, sim_weight=0.0):
    """Build prediction for a round using given blend weight and score it.

    Args:
        sim_weight: If > 0, blend Monte Carlo simulator predictions between L6 and L7.
    """
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

        # Layer 3: Fill unobserved dynamic (with empty obs to fill everything)
        tensor = fill_unobserved_dynamic(tensor, grid, settlements, [], seed)

        # Layer 6: GBT blend (uniform blend weight as requested)
        gbt_pred = gbt_predict_with_models(gbt_models, grid, settlements)
        if gbt_pred is not None:
            h, w, _ = tensor.shape
            for y in range(h):
                for x in range(w):
                    code = grid[y][x]
                    if code in {10, 5}:
                        continue
                    tensor[y, x] = (1 - blend_weight) * tensor[y, x] + blend_weight * gbt_pred[y, x]

        # Layer 6.5: Simulator ensemble (only when sim_weight > 0)
        if sim_weight > 0:
            try:
                sim_params = fit_hidden_params(grid, settlements, all_observations)
                sim_probs = simulate_monte_carlo(
                    grid, settlements, n_sims=300, **sim_params
                )
                h, w, _ = tensor.shape
                for y in range(h):
                    for x in range(w):
                        code = grid[y][x]
                        if code in {10, 5}:
                            continue
                        tensor[y, x] = (1 - sim_weight) * tensor[y, x] + sim_weight * sim_probs[y, x]
            except Exception:
                pass  # gracefully fall back to no blending

        # Layer 7: Global observation ratio correction
        same_seed_obs = [o for o in all_observations if o.get("seed_index") == seed]
        all_seed_obs = all_observations  # use all observations for L7
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

        # Normalize
        tensor = normalize_prediction(tensor)

        # Score
        score = compute_score(tensor, gt)
        scores.append(score)

    return np.mean(scores)


def main():
    blend_values = [0.40]
    test_rounds = [1, 2, 4, 5, 6]

    # Header
    print(f"{'Blend':<8}", end="")
    for r in test_rounds:
        print(f"{'R' + str(r):>8}", end="")
    print(f"{'AVG':>8}")
    print("-" * 48)

    for blend in blend_values:
        round_scores = []
        for held_out in test_rounds:
            train_on = [r for r in test_rounds if r != held_out]
            print(f"  Training on R{train_on}, testing on R{held_out}...", flush=True)
            gbt_models = train_gbt_models(train_on)
            score = evaluate_round_with_blend(held_out, gbt_models, blend)
            round_scores.append(score)
            print(f"    -> {score:.2f}", flush=True)

        avg = np.mean(round_scores)
        print(f"{blend:<8.2f}", end="")
        for s in round_scores:
            print(f"{s:>8.2f}", end="")
        print(f"{avg:>8.2f}")
        print()


if __name__ == "__main__":
    main()
