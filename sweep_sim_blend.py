"""Sweep SIM_BLEND_WEIGHT in LORO cross-validation.

Finds the optimal blend weight for the simulator ensemble.
Uses the same LORO framework as loro_blend_cv.py but sweeps the sim weight.
"""

import json
import os
import sys
import numpy as np
import warnings
import time

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

# L7 strengths (from current best model)
L7_STRENGTHS = np.array([1.38, 0.90, 0.44, 0.61, 1.16, 0.0])

GBT_BLEND = 0.30  # fixed from previous optimization

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


def train_gbt_models(train_rounds):
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
                n_estimators=200, max_depth=4, learning_rate=0.1,
                reg_alpha=0.1, reg_lambda=2.0, subsample=0.9,
                colsample_bytree=0.9, min_child_weight=3,
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


def evaluate_round(test_round_num, gbt_models, sim_weight, n_sims=300):
    round_id = ROUNDS[test_round_num]
    initial_states, gts = load_round_data(test_round_num)
    all_observations = load_observations(round_id)

    scores = []
    for seed in range(5):
        grid = initial_states[seed]["grid"]
        settlements = initial_states[seed]["settlements"]
        gt = gts[seed]

        # Layer 1: Static
        tensor = build_static_prediction(grid)

        # Layer 3: Fill unobserved
        tensor = fill_unobserved_dynamic(tensor, grid, settlements, [], seed)

        # Layer 6: GBT blend
        gbt_pred = gbt_predict_with_models(gbt_models, grid, settlements)
        if gbt_pred is not None:
            grid_arr = np.array(grid)
            dynamic = ~np.isin(grid_arr, [10, 5])
            tensor[dynamic] = (1 - GBT_BLEND) * tensor[dynamic] + GBT_BLEND * gbt_pred[dynamic]

        # Layer 6.5: Simulator ensemble
        if sim_weight > 0:
            try:
                sim_params = fit_hidden_params(grid, settlements, all_observations)
                sim_probs = simulate_monte_carlo(
                    grid, settlements, n_sims=n_sims, **sim_params
                )
                grid_arr = np.array(grid)
                dynamic = ~np.isin(grid_arr, [10, 5])
                tensor[dynamic] = (1 - sim_weight) * tensor[dynamic] + sim_weight * sim_probs[dynamic]
            except Exception as e:
                print(f"    Sim failed for R{test_round_num} seed{seed}: {e}")

        # Layer 7: Observation ratio correction
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


def main():
    sim_weights = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    test_rounds = [1, 2, 4, 5]

    print(f"{'SimW':<8}", end="")
    for r in test_rounds:
        print(f"{'R' + str(r):>8}", end="")
    print(f"{'AVG':>8}  Time")
    print("-" * 60)

    for sw in sim_weights:
        t0 = time.time()
        round_scores = []
        for held_out in test_rounds:
            train_on = [r for r in test_rounds if r != held_out]
            gbt_models = train_gbt_models(train_on)
            avg_score, per_seed = evaluate_round(held_out, gbt_models, sw, n_sims=300)
            round_scores.append(avg_score)
            print(f"  R{held_out}: {avg_score:.2f} {[f'{s:.1f}' for s in per_seed]}", flush=True)

        avg = np.mean(round_scores)
        elapsed = time.time() - t0
        print(f"{sw:<8.2f}", end="")
        for s in round_scores:
            print(f"{s:>8.2f}", end="")
        print(f"{avg:>8.2f}  {elapsed:.0f}s")
        print()


if __name__ == "__main__":
    main()
