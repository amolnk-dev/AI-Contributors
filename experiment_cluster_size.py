"""Experiment: settlement_cluster_size feature.

Hypothesis: large settlement clusters (connected through nearby settlements)
behave differently from isolated settlements. Large clusters have more
internal trade and faction consolidation; small isolated settlements are more
vulnerable to winter.

Implementation:
1. BFS from each settlement, connecting through nearby settlements (dist<=threshold).
2. Each cell gets the cluster_size of its nearest settlement's cluster.
3. Test correlation with GT across thresholds, then LORO CV with best.

Evaluation: Leave-one-round-out cross-validation on R1, R2, R4, R5.
"""

import json
import os
import sys
import time
from collections import deque

import numpy as np
import warnings

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
from dtos import NUM_CLASSES, PROB_FLOOR, TERRAIN_TO_CLASS

# Round info
ROUNDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
}

L7_STRENGTHS = np.array([1.38, 0.90, 0.44, 0.61, 1.16, 0.0])

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ──────────────────────────────────────────────────────────────
# Settlement cluster computation
# ──────────────────────────────────────────────────────────────

def compute_settlement_clusters(initial_grid, settlements, threshold=5):
    """Compute settlement cluster sizes via BFS.

    Settlements are connected if they are within Manhattan distance <= threshold.
    Returns:
      - cluster_size_grid: H x W array, each cell gets the cluster_size
        of its nearest settlement's cluster.
      - settlement_cluster_map: dict mapping (sy, sx) -> cluster_size
    """
    h = len(initial_grid)
    w = len(initial_grid[0]) if h > 0 else 0

    # Get settlement positions
    settl_pos = []
    for s in settlements:
        if isinstance(s, dict):
            sx, sy = s.get("x", 0), s.get("y", 0)
        else:
            sx, sy = s.x, s.y
        settl_pos.append((sy, sx))  # (y, x) format

    if not settl_pos:
        return np.zeros((h, w), dtype=np.int32), {}

    # Build adjacency graph: settlements within Manhattan dist <= threshold
    n = len(settl_pos)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dy = abs(settl_pos[i][0] - settl_pos[j][0])
            dx = abs(settl_pos[i][1] - settl_pos[j][1])
            if dy + dx <= threshold:
                adj[i].append(j)
                adj[j].append(i)

    # BFS to find connected components
    visited = [False] * n
    cluster_id = [-1] * n
    cluster_sizes = {}
    cid = 0
    for start in range(n):
        if visited[start]:
            continue
        q = deque([start])
        visited[start] = True
        cluster_id[start] = cid
        members = [start]
        while q:
            node = q.popleft()
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    cluster_id[nb] = cid
                    members.append(nb)
                    q.append(nb)
        cluster_sizes[cid] = len(members)
        cid += 1

    # Map: settlement (y, x) -> cluster_size
    settlement_cluster_map = {}
    for i, (sy, sx) in enumerate(settl_pos):
        settlement_cluster_map[(sy, sx)] = cluster_sizes[cluster_id[i]]

    # For each non-static cell, find nearest settlement and assign its cluster_size
    cluster_size_grid = np.zeros((h, w), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            if initial_grid[y][x] in {10, 5}:
                continue
            best_dist = 999
            best_csize = 0
            for i, (sy, sx) in enumerate(settl_pos):
                d = abs(y - sy) + abs(x - sx)
                if d < best_dist:
                    best_dist = d
                    best_csize = cluster_sizes[cluster_id[i]]
            cluster_size_grid[y, x] = best_csize

    return cluster_size_grid, settlement_cluster_map


def extract_features_with_cluster(grid, settlements, threshold=5):
    """Extract cell features + settlement_cluster_size appended."""
    base_feats, coords = _extract_cell_features(grid, settlements)
    if len(base_feats) == 0:
        return base_feats, coords

    cluster_grid, _ = compute_settlement_clusters(grid, settlements, threshold=threshold)

    # Append cluster_size for each dynamic cell
    cluster_col = np.array([cluster_grid[y, x] for y, x in coords]).reshape(-1, 1)
    augmented = np.hstack([base_feats, cluster_col])
    return augmented, coords


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

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
# Correlation analysis
# ──────────────────────────────────────────────────────────────

def analyze_correlation_sweep():
    """Compute correlation between settlement_cluster_size and GT for various thresholds."""
    print("=" * 60)
    print("CORRELATION ANALYSIS: settlement_cluster_size vs GT")
    print("=" * 60)

    for threshold in [3, 4, 5, 6, 7, 8, 10]:
        all_cluster_sizes = []
        all_gt_probs = {i: [] for i in range(NUM_CLASSES)}
        all_terrain_codes = []

        for rnum in [1, 2, 4, 5]:
            initial_states, gts = load_round_data(rnum)

            for seed in range(5):
                grid = initial_states[seed]["grid"]
                settlements = initial_states[seed]["settlements"]
                gt = gts[seed]

                cluster_grid, _ = compute_settlement_clusters(grid, settlements, threshold=threshold)

                h, w = len(grid), len(grid[0])
                for y in range(h):
                    for x in range(w):
                        code = grid[y][x]
                        if code in {10, 5}:  # skip static
                            continue
                        all_cluster_sizes.append(cluster_grid[y, x])
                        for cls in range(NUM_CLASSES):
                            all_gt_probs[cls].append(gt[y, x, cls])
                        all_terrain_codes.append(code)

        cs = np.array(all_cluster_sizes)
        unique_vals = np.unique(cs)
        n_unique = len(unique_vals)

        if cs.std() == 0:
            print(f"\n  threshold={threshold}: NO VARIANCE (all size={cs[0]})")
            continue

        corrs = {}
        class_names = ["empty", "settl", "port", "ruin", "forest", "mount"]
        for cls in range(NUM_CLASSES):
            gt_arr = np.array(all_gt_probs[cls])
            if gt_arr.std() > 0:
                corrs[cls] = np.corrcoef(cs, gt_arr)[0, 1]
            else:
                corrs[cls] = 0.0

        print(f"\n  threshold={threshold}: {n_unique} unique values, range={cs.min()}-{cs.max()}, mean={cs.mean():.1f}")
        print(f"    Correlations: " + ", ".join(f"{class_names[c]}={corrs[c]:+.4f}" for c in range(5)))

        # Show settlement cells specifically
        terrain_arr = np.array(all_terrain_codes)
        for code, name in [(1, "Settlement"), (4, "Forest"), (11, "Plains")]:
            mask = terrain_arr == code
            if mask.sum() < 10 and cs[mask].std() > 0:
                continue
            sub_cs = cs[mask]
            if sub_cs.std() == 0:
                continue
            sub_gt_settl = np.array(all_gt_probs[1])[mask]
            r = np.corrcoef(sub_cs, sub_gt_settl)[0, 1]
            print(f"    {name} cells: corr(cluster_sz, P(settl))={r:+.4f}, n={mask.sum()}")

    # Detailed stratified analysis for best threshold
    print("\n\nDETAILED ANALYSIS (threshold=5):")
    best_thresh = 5
    all_cs = []
    all_gt = []
    all_terrain = []
    for rnum in [1, 2, 4, 5]:
        initial_states, gts = load_round_data(rnum)
        for seed in range(5):
            grid = initial_states[seed]["grid"]
            settlements = initial_states[seed]["settlements"]
            gt = gts[seed]
            cluster_grid, _ = compute_settlement_clusters(grid, settlements, threshold=best_thresh)
            h, w = len(grid), len(grid[0])
            for y in range(h):
                for x in range(w):
                    code = grid[y][x]
                    if code in {10, 5}:
                        continue
                    all_cs.append(cluster_grid[y, x])
                    all_gt.append(gt[y, x])
                    all_terrain.append(code)

    cs = np.array(all_cs)
    gt = np.array(all_gt)
    terrain = np.array(all_terrain)

    print(f"\nMean GT probabilities by cluster_size (all dynamic cells):")
    print(f"  {'ClustSz':<10} {'n':>6} {'P(empty)':>10} {'P(settl)':>10} {'P(port)':>10} {'P(ruin)':>10} {'P(forest)':>10}")
    for v in sorted(np.unique(cs)):
        mask = cs == v
        n = mask.sum()
        if n < 20:
            continue
        probs = gt[mask].mean(axis=0)
        print(f"  {v:<10} {n:>6} {probs[0]:>10.4f} {probs[1]:>10.4f} {probs[2]:>10.4f} {probs[3]:>10.4f} {probs[4]:>10.4f}")

    # Settlement cells only
    mask_settl = terrain == 1
    print(f"\nSettlement cells only (n={mask_settl.sum()}):")
    print(f"  {'ClustSz':<10} {'n':>6} {'P(empty)':>10} {'P(settl)':>10} {'P(ruin)':>10} {'P(forest)':>10}")
    for v in sorted(np.unique(cs[mask_settl])):
        m = mask_settl & (cs == v)
        n = m.sum()
        if n < 5:
            continue
        probs = gt[m].mean(axis=0)
        print(f"  {v:<10} {n:>6} {probs[0]:>10.4f} {probs[1]:>10.4f} {probs[3]:>10.4f} {probs[4]:>10.4f}")


# ──────────────────────────────────────────────────────────────
# LORO CV
# ──────────────────────────────────────────────────────────────

def train_terrain_xgb(X_data, Y_data):
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


def predict_with_models(models, features, coords, grid, h, w):
    """Generate H x W x 6 tensor from terrain-specific models."""
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

        models_t = models.get(ttype)
        if models_t is None:
            continue

        pred_v = np.zeros(NUM_CLASSES)
        for cls in range(NUM_CLASSES):
            pred_v[cls] = models_t[cls].predict(features[i:i + 1])[0]
        tensor[y, x] = np.maximum(pred_v, PROB_FLOOR)
        tensor[y, x] /= tensor[y, x].sum()

    return tensor


def loro_cv(use_cluster_feature=False, cluster_threshold=5, blend_weight=0.30):
    """Leave-one-round-out CV.

    Args:
        use_cluster_feature: if True, append settlement_cluster_size to features
        cluster_threshold: Manhattan distance threshold for clustering settlements
        blend_weight: GBT blend weight
    """
    test_rounds = [1, 2, 4, 5]
    round_data = {}
    for rnum in test_rounds:
        round_data[rnum] = load_round_data(rnum)

    round_scores = []
    for held_out in test_rounds:
        train_on = [r for r in test_rounds if r != held_out]

        # Collect training data
        X_data = {"plains": [], "forest": [], "settl": []}
        Y_data = {"plains": [], "forest": [], "settl": []}

        for rnum in train_on:
            initial_states, gts = round_data[rnum]
            for seed in range(5):
                grid = initial_states[seed]["grid"]
                settlements = initial_states[seed]["settlements"]
                gt = gts[seed]

                if use_cluster_feature:
                    feats, coords = extract_features_with_cluster(grid, settlements, threshold=cluster_threshold)
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
        models = train_terrain_xgb(X_data, Y_data)

        # Evaluate on held-out round
        round_id = ROUNDS[held_out]
        initial_states, gts = round_data[held_out]
        all_observations = load_observations(round_id)

        seeds_scores = []
        for seed in range(5):
            grid = initial_states[seed]["grid"]
            settlements = initial_states[seed]["settlements"]
            gt = gts[seed]
            h, w = len(grid), len(grid[0])

            if use_cluster_feature:
                feats, coords = extract_features_with_cluster(grid, settlements, threshold=cluster_threshold)
            else:
                feats, coords = _extract_cell_features(grid, settlements)

            # Layer 1 + 3
            tensor = build_static_prediction(grid)
            tensor = fill_unobserved_dynamic(tensor, grid, settlements, [], seed)

            # Layer 6: GBT blend
            gbt_pred = predict_with_models(models, feats, coords, grid, h, w)
            for y in range(h):
                for x in range(w):
                    code = grid[y][x]
                    if code in {10, 5}:
                        continue
                    tensor[y, x] = (1 - blend_weight) * tensor[y, x] + blend_weight * gbt_pred[y, x]

            # Layer 7: observation ratio correction
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
            seeds_scores.append(score)

        round_avg = np.mean(seeds_scores)
        round_scores.append(round_avg)
        print(f"    R{held_out}: {round_avg:.2f} (seeds: {' '.join(f'{s:.2f}' for s in seeds_scores)})")

    return round_scores


def main():
    os.chdir(os.path.dirname(__file__))

    # Step 1: Correlation analysis across thresholds
    analyze_correlation_sweep()

    # Step 2: LORO CV comparison
    print("\n\n" + "=" * 60)
    print("LORO CV: Baseline vs + settlement_cluster_size")
    print("=" * 60)

    test_rounds = [1, 2, 4, 5]

    print("\nBaseline (no cluster feature):")
    t0 = time.time()
    baseline_scores = loro_cv(use_cluster_feature=False, blend_weight=0.30)
    baseline_avg = np.mean(baseline_scores)
    print(f"  Time: {time.time()-t0:.1f}s\n")

    results = [("Baseline (30 feats)", baseline_scores, baseline_avg)]

    for thresh in [4, 5, 6, 8]:
        print(f"+cluster_size (threshold={thresh}):")
        t0 = time.time()
        scores = loro_cv(use_cluster_feature=True, cluster_threshold=thresh, blend_weight=0.30)
        avg = np.mean(scores)
        print(f"  Time: {time.time()-t0:.1f}s\n")
        results.append((f"+cluster_sz (t={thresh})", scores, avg))

    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Approach':<30}", end="")
    for r in test_rounds:
        print(f"{'R' + str(r):>8}", end="")
    print(f"{'AVG':>8}  {'Delta':>8}")
    print("-" * 78)

    for name, scores, avg in results:
        delta = avg - baseline_avg
        print(f"{name:<30}", end="")
        for s in scores:
            print(f"{s:>8.2f}", end="")
        print(f"{avg:>8.2f}  {delta:>+8.4f}")


if __name__ == "__main__":
    main()
