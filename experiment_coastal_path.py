"""Experiment: coastal_path_distance — BFS distance to nearest ocean over passable terrain.

Hypothesis: Manhattan dist_ocean ignores mountains & other ocean bodies blocking the path.
BFS coastal_path_distance walks only through passable (non-ocean, non-mountain) cells,
so it captures whether a cell can actually REACH the coast. Settlements that can reach
the coast are more likely to develop ports.

Steps:
  1. Compute BFS coastal_path_distance for all dynamic cells
  2. Correlate with GT P(port) across all 5 rounds × 5 seeds
  3. Compare with existing dist_ocean (Manhattan)
  4. If promising, integrate into feature vector and run LORO
"""

import json
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from dtos import NUM_CLASSES, PROB_FLOOR, TERRAIN_TO_CLASS
from evaluate import compute_score
from utils import normalize_prediction

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

ROUNDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
}


def load_round_data(round_num):
    with open(os.path.join(DATA_DIR, f"round{round_num}_initial.json")) as f:
        info = json.load(f)
    initial_states = info["initial_states"]
    gts = []
    for seed in range(5):
        gt_path = os.path.join(DATA_DIR, f"gt_r{round_num}_seed{seed}.npy")
        gts.append(np.load(gt_path))
    return initial_states, gts


def compute_coastal_path_distance(grid):
    """BFS distance from each cell to nearest ocean, walking only passable terrain.

    Passable = not ocean (10) and not mountain (5).
    Returns H×W array with distances (99 = unreachable).
    """
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    grid_arr = np.array(grid)

    # Start BFS from all cells adjacent to ocean (the "coast")
    # These are passable cells that touch ocean
    dist = np.full((h, w), 99, dtype=np.int32)
    q = deque()

    for y in range(h):
        for x in range(w):
            code = grid[y][x]
            if code in {10, 5}:
                continue  # skip ocean and mountain
            # Check if this passable cell is adjacent to ocean
            is_coastal = False
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and grid[ny][nx] == 10:
                    is_coastal = True
                    break
            if is_coastal:
                dist[y, x] = 0
                q.append((y, x))

    # BFS over passable terrain
    while q:
        cy, cx = q.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w:
                code = grid[ny][nx]
                if code not in {10, 5} and dist[ny, nx] > dist[cy, cx] + 1:
                    dist[ny, nx] = dist[cy, cx] + 1
                    q.append((ny, nx))

    return dist


def compute_manhattan_ocean_dist(grid):
    """Manhattan distance to nearest ocean (scan radius 8), matching model.py."""
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    result = np.full((h, w), 99, dtype=np.int32)

    for y in range(h):
        for x in range(w):
            if grid[y][x] in {10, 5}:
                continue
            d_ocean = 99
            for dy in range(-8, 9):
                for dx in range(-8, 9):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        d = abs(dy) + abs(dx)
                        if d > 8:
                            continue
                        if grid[ny][nx] == 10 and d < d_ocean:
                            d_ocean = d
            result[y, x] = d_ocean

    return result


def analyze_correlation():
    """Compute correlation between coastal_path_distance and GT P(port)."""

    all_cpd = []  # coastal path distance
    all_manhattan = []  # manhattan dist to ocean
    all_port_prob = []  # GT P(port)
    all_settl_prob = []  # GT P(settlement)
    all_codes = []  # initial terrain code

    for rnum in ROUNDS:
        initial_states, gts = load_round_data(rnum)
        for seed in range(5):
            grid = initial_states[seed]["grid"]
            gt = gts[seed]
            h, w = len(grid), len(grid[0])

            cpd = compute_coastal_path_distance(grid)
            manhattan = compute_manhattan_ocean_dist(grid)

            for y in range(h):
                for x in range(w):
                    code = grid[y][x]
                    if code in {10, 5}:
                        continue
                    all_cpd.append(cpd[y, x])
                    all_manhattan.append(manhattan[y, x])
                    all_port_prob.append(gt[y, x, 2])  # class 2 = port
                    all_settl_prob.append(gt[y, x, 1])  # class 1 = settlement
                    all_codes.append(code)

    all_cpd = np.array(all_cpd)
    all_manhattan = np.array(all_manhattan)
    all_port_prob = np.array(all_port_prob)
    all_settl_prob = np.array(all_settl_prob)
    all_codes = np.array(all_codes)

    print("=" * 60)
    print("COASTAL PATH DISTANCE vs GT P(port) — Correlation Analysis")
    print("=" * 60)
    print(f"Total dynamic cells: {len(all_cpd)}")
    print()

    # Overall correlation
    mask = all_port_prob > 0  # cells that have any port probability
    print(f"Cells with P(port) > 0: {mask.sum()}")

    # Pearson correlation
    from scipy.stats import pearsonr, spearmanr

    print("\n--- All dynamic cells ---")
    r_cpd, p_cpd = pearsonr(all_cpd, all_port_prob)
    r_man, p_man = pearsonr(all_manhattan, all_port_prob)
    print(f"coastal_path_dist vs P(port): r={r_cpd:.4f} (p={p_cpd:.2e})")
    print(f"manhattan_ocean   vs P(port): r={r_man:.4f} (p={p_man:.2e})")

    sr_cpd, sp_cpd = spearmanr(all_cpd, all_port_prob)
    sr_man, sp_man = spearmanr(all_manhattan, all_port_prob)
    print(f"coastal_path_dist vs P(port): rho={sr_cpd:.4f} (Spearman)")
    print(f"manhattan_ocean   vs P(port): rho={sr_man:.4f} (Spearman)")

    # Breakdown by distance bucket
    print("\n--- P(port) by coastal_path_distance bucket ---")
    for d in range(10):
        mask_d = all_cpd == d
        if mask_d.sum() > 0:
            print(f"  cpd={d:2d}: n={mask_d.sum():6d}  mean_P(port)={all_port_prob[mask_d].mean():.5f}  "
                  f"mean_P(settl)={all_settl_prob[mask_d].mean():.5f}")
    mask_far = all_cpd >= 10
    if mask_far.sum() > 0:
        print(f"  cpd>=10: n={mask_far.sum():6d}  mean_P(port)={all_port_prob[mask_far].mean():.5f}  "
              f"mean_P(settl)={all_settl_prob[mask_far].mean():.5f}")
    mask_unreach = all_cpd >= 99
    if mask_unreach.sum() > 0:
        print(f"  cpd=99 (unreachable): n={mask_unreach.sum():6d}  mean_P(port)={all_port_prob[mask_unreach].mean():.5f}")

    print("\n--- P(port) by Manhattan ocean_dist bucket ---")
    for d in range(10):
        mask_d = all_manhattan == d
        if mask_d.sum() > 0:
            print(f"  man={d:2d}: n={mask_d.sum():6d}  mean_P(port)={all_port_prob[mask_d].mean():.5f}  "
                  f"mean_P(settl)={all_settl_prob[mask_d].mean():.5f}")
    mask_far = all_manhattan >= 10
    if mask_far.sum() > 0:
        print(f"  man>=10: n={mask_far.sum():6d}  mean_P(port)={all_port_prob[mask_far].mean():.5f}")

    # Key comparison: cells where cpd != manhattan (i.e., where path matters)
    diff_mask = all_cpd != all_manhattan
    print(f"\n--- Cells where cpd != manhattan: {diff_mask.sum()} ({100*diff_mask.mean():.1f}%) ---")
    if diff_mask.sum() > 0:
        print(f"  These cells: mean_P(port)={all_port_prob[diff_mask].mean():.5f}")
        # Among these, which better predicts port?
        # Lower distance should correlate with higher P(port) for port prediction
        both_diff = diff_mask & (all_port_prob > 0)
        if both_diff.sum() > 0:
            r_cpd_d, _ = pearsonr(all_cpd[both_diff], all_port_prob[both_diff])
            r_man_d, _ = pearsonr(all_manhattan[both_diff], all_port_prob[both_diff])
            print(f"  Among P(port)>0 cells: cpd vs P(port) r={r_cpd_d:.4f}, manhattan r={r_man_d:.4f}")

    # Added info: cpd for cells that are coastal (cpd=0)
    coastal = all_cpd == 0
    print(f"\n--- Coastal cells (cpd=0): {coastal.sum()} ---")
    print(f"  mean P(port)={all_port_prob[coastal].mean():.5f}")
    print(f"  mean P(settl)={all_settl_prob[coastal].mean():.5f}")

    # Non-coastal cells that manhattan says are close to ocean
    non_coastal_close = (all_cpd > 0) & (all_manhattan <= 2)
    print(f"\n--- Non-coastal but manhattan<=2: {non_coastal_close.sum()} ---")
    if non_coastal_close.sum() > 0:
        print(f"  mean P(port)={all_port_prob[non_coastal_close].mean():.5f}")

    # Cells with path blocked (cpd > manhattan + 2) — mountains creating detours
    blocked = all_cpd > all_manhattan + 2
    print(f"\n--- Path blocked (cpd > manhattan+2): {blocked.sum()} ---")
    if blocked.sum() > 0:
        print(f"  mean P(port)={all_port_prob[blocked].mean():.5f}")
        print(f"  mean cpd={all_cpd[blocked].mean():.1f}, mean manhattan={all_manhattan[blocked].mean():.1f}")

    return all_cpd, all_manhattan, all_port_prob


def run_loro_with_coastal_path():
    """Run LORO with coastal_path_distance added to the feature vector.

    Compare baseline (30 features) vs augmented (31 features with cpd).
    """
    import warnings
    warnings.filterwarnings("ignore")
    import xgboost as xgb
    from model import (
        _extract_cell_features,
        build_static_prediction,
        fill_unobserved_dynamic,
    )
    from utils import load_observations

    L7_STRENGTHS = np.array([1.38, 0.90, 0.44, 0.61, 1.16, 0.0])
    test_rounds = [1, 2, 4, 5]
    BLEND_WEIGHT = 0.30  # current production value

    def extract_features_with_cpd(grid, settlements):
        """Extract standard features + append coastal_path_distance."""
        feats, coords = _extract_cell_features(grid, settlements)
        if len(feats) == 0:
            return feats, coords

        cpd = compute_coastal_path_distance(grid)
        cpd_col = np.array([cpd[y, x] for y, x in coords], dtype=np.float64).reshape(-1, 1)
        feats_aug = np.hstack([feats, cpd_col])
        return feats_aug, coords

    def train_gbt(train_rounds, use_cpd=False):
        X_data = {"plains": [], "forest": [], "settl": []}
        Y_data = {"plains": [], "forest": [], "settl": []}

        for rnum in train_rounds:
            initial_states, gts = load_round_data(rnum)
            for seed in range(5):
                grid = initial_states[seed]["grid"]
                settlements = initial_states[seed]["settlements"]
                gt = gts[seed]

                if use_cpd:
                    feats, coords = extract_features_with_cpd(grid, settlements)
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
        for ttype in ["plains", "forest", "settl"]:
            X = np.array(X_data[ttype])
            Y = np.array(Y_data[ttype])
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
            models[ttype] = terrain_models
        return models

    def gbt_predict_local(models_dict, grid, settlements, use_cpd=False):
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        if use_cpd:
            features, coords = extract_features_with_cpd(grid, settlements)
        else:
            features, coords = _extract_cell_features(grid, settlements)

        if len(features) == 0:
            return None

        tensor = np.zeros((h, w, NUM_CLASSES))
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

            models = models_dict.get(ttype)
            if models is None:
                continue

            pred_v = np.zeros(NUM_CLASSES)
            for cls in range(NUM_CLASSES):
                pred_v[cls] = models[cls].predict(features[i:i + 1])[0]
            tensor[y, x] = np.maximum(pred_v, PROB_FLOOR)
            tensor[y, x] /= tensor[y, x].sum()

        return tensor

    def eval_round(test_rnum, gbt_models, use_cpd=False):
        round_id = ROUNDS[test_rnum]
        initial_states, gts = load_round_data(test_rnum)
        all_observations = load_observations(round_id)

        scores = []
        for seed in range(5):
            grid = initial_states[seed]["grid"]
            settlements = initial_states[seed]["settlements"]
            gt = gts[seed]

            tensor = build_static_prediction(grid)
            tensor = fill_unobserved_dynamic(tensor, grid, settlements, [], seed)

            gbt_pred = gbt_predict_local(gbt_models, grid, settlements, use_cpd)
            if gbt_pred is not None:
                h, w, _ = tensor.shape
                for y in range(h):
                    for x in range(w):
                        if grid[y][x] not in {10, 5}:
                            tensor[y, x] = (1 - BLEND_WEIGHT) * tensor[y, x] + BLEND_WEIGHT * gbt_pred[y, x]

            # L7
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

        return np.mean(scores)

    # Run LORO for both baseline and +cpd
    print("\n" + "=" * 60)
    print("LORO: Baseline vs +coastal_path_distance")
    print("=" * 60)

    header = f"{'Config':<20}"
    for r in test_rounds:
        header += f"{'R' + str(r):>8}"
    header += f"{'AVG':>8}"
    print(header)
    print("-" * 56)

    for label, use_cpd in [("baseline (30 feat)", False), ("+cpd (31 feat)", True)]:
        round_scores = []
        for held_out in test_rounds:
            train_on = [r for r in test_rounds if r != held_out]
            print(f"  {label}: train={train_on}, test=R{held_out}...", flush=True)
            models = train_gbt(train_on, use_cpd=use_cpd)
            score = eval_round(held_out, models, use_cpd=use_cpd)
            round_scores.append(score)
            print(f"    -> {score:.2f}", flush=True)

        avg = np.mean(round_scores)
        line = f"{label:<20}"
        for s in round_scores:
            line += f"{s:>8.2f}"
        line += f"{avg:>8.2f}"
        print(line)
        print()


if __name__ == "__main__":
    analyze_correlation()
    run_loro_with_coastal_path()
