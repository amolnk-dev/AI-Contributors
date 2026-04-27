"""Astar Island prediction engine — heuristic + GBT hybrid model.

Prediction layers:
  1. STATIC — distance-based coastal/inland priors from GT tables
  2. OBSERVED — DISABLED (noisy with few observations)
  3. CONTEXT — adj_forests, adj_settlements, exposed_coastal priors
  4. CROSS-SEED — DISABLED
  5. CALIBRATION — blend with learned context priors
  6. GBT BLEND — blend with gradient-boosted tree predictions
  6.5 SIMULATOR ENSEMBLE — Monte Carlo simulator blended with XGBoost+heuristic
  7. OBSERVATION CORRECTION — ratio correction from pooled observations

The GBT model captures feature interactions the hand-tuned tables miss.
Trained on Round 1 GT, validated with leave-one-seed-out CV.
"""

import json
import os
import pickle
import warnings
from typing import Optional

import numpy as np
from scipy import ndimage

warnings.filterwarnings("ignore", category=UserWarning)  # suppress xgboost warnings

from dtos import (
    TERRAIN_TO_CLASS,
    NUM_CLASSES,
    STATIC_TERRAIN_CODES,
    PROB_FLOOR,
)
from utils import normalize_prediction, grid_to_class_array
from simulator import simulate_monte_carlo, fit_hidden_params


# ──────────────────────────────────────────────────────────────
# GBT model support
# ──────────────────────────────────────────────────────────────

GBT_BLEND_WEIGHT = 0.35  # fallback for unknown terrain codes
SIM_BLEND_WEIGHT = 0.0  # disabled — proven +0.00 LORO, adds 30s latency
_gbt_models = None  # lazy-loaded

# Import shared params — single source of truth (see params.py)
from params import (
    BLEND_TERRAIN, ROUND_TYPE_THRESHOLDS, ROUND_TYPE_L7,
    L7_MIN_OBS, EMP_BLEND_EXPANSION, EMP_BLEND_NORMAL,
    classify_round_type,
)


def _extract_cell_features(
    initial_grid: list[list[int]],
    settlements: list,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Extract per-cell feature vectors for dynamic cells.

    Returns (features_array, list_of_coords) where coords maps row index → (y, x).
    """
    h = len(initial_grid)
    w = len(initial_grid[0]) if h > 0 else 0

    settl_pos = []
    port_pos = []
    for s in settlements:
        if isinstance(s, dict):
            settl_pos.append((s.get("x", 0), s.get("y", 0)))
            if s.get("has_port"):
                port_pos.append((s.get("x", 0), s.get("y", 0)))
        else:
            settl_pos.append((s.x, s.y))
            if getattr(s, "has_port", False):
                port_pos.append((s.x, s.y))

    # Pre-compute grid-level features
    grid_arr = np.array(initial_grid)
    land_mask = (grid_arr != 10) & (grid_arr != 5)
    passable_mask = land_mask.astype(np.float64)

    # Connected components
    labeled, _ = ndimage.label(land_mask)
    comp_sizes = {}
    comp_settl = {}
    for y_ in range(h):
        for x_ in range(w):
            comp = labeled[y_, x_]
            if comp > 0:
                comp_sizes[comp] = comp_sizes.get(comp, 0) + 1
                if initial_grid[y_][x_] in {1, 2}:
                    comp_settl[comp] = comp_settl.get(comp, 0) + 1

    # BFS distance from all settlements over passable terrain
    from collections import deque
    bfs_dist = np.full((h, w), 99, dtype=np.int32)
    q = deque()
    for sx, sy in settl_pos:
        if 0 <= sy < h and 0 <= sx < w:
            bfs_dist[sy, sx] = 0
            q.append((sy, sx))
    while q:
        cy, cx = q.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and land_mask[ny, nx] and bfs_dist[ny, nx] > bfs_dist[cy, cx] + 1:
                bfs_dist[ny, nx] = bfs_dist[cy, cx] + 1
                q.append((ny, nx))

    # Passable cells in 5x5 window (convolution)
    kernel5 = np.ones((5, 5))
    passable_r2 = ndimage.convolve(passable_mask, kernel5, mode='constant', cval=0.0)

    # Land ratio in 7x7 window (radius 3)
    land_float = (grid_arr != 10).astype(np.float64)
    kernel7 = np.ones((7, 7))
    land_sum_r3 = ndimage.convolve(land_float, kernel7, mode='constant', cval=0.0)
    land_count_r3 = ndimage.convolve(np.ones((h, w)), kernel7, mode='constant', cval=0.0)
    land_ratio_r3 = land_sum_r3 / np.maximum(land_count_r3, 1.0)

    # Precompute adjacency counts via convolution (replaces per-cell 8-neighbor loops)
    kernel3 = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.float64)
    adj_ocean_arr = ndimage.convolve((grid_arr == 10).astype(np.float64), kernel3, mode='constant', cval=0.0)
    adj_forest_arr = ndimage.convolve((grid_arr == 4).astype(np.float64), kernel3, mode='constant', cval=0.0)
    adj_settl_arr = ndimage.convolve(np.isin(grid_arr, [1, 2]).astype(np.float64), kernel3, mode='constant', cval=0.0)
    adj_mountain_arr = ndimage.convolve((grid_arr == 5).astype(np.float64), kernel3, mode='constant', cval=0.0)

    # Precompute distance to nearest ocean and mountain via distance transform
    # distance_transform_cdt computes Manhattan (chessboard) distance to nearest zero
    ocean_mask = (grid_arr == 10)
    mountain_mask = (grid_arr == 5)
    # distance_transform gives distance to nearest 0 — we invert so terrain=0, non-terrain=1
    dist_ocean_arr = ndimage.distance_transform_cdt(~ocean_mask, metric='taxicab').astype(np.int32)
    dist_ocean_arr = np.minimum(dist_ocean_arr, 99)
    if mountain_mask.any():
        dist_mountain_arr = ndimage.distance_transform_cdt(~mountain_mask, metric='taxicab').astype(np.int32)
        dist_mountain_arr = np.minimum(dist_mountain_arr, 99)
    else:
        dist_mountain_arr = np.full((h, w), 99, dtype=np.int32)

    # Precompute forests in radius 2 via diamond kernel convolution
    forest_float = (grid_arr == 4).astype(np.float64)
    kernel_diamond = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,0,1,1],[0,1,1,1,0],[0,0,1,0,0]], dtype=np.float64)
    forests_r2_arr = ndimage.convolve(forest_float, kernel_diamond, mode='constant', cval=0.0)

    # Precompute settlement distance arrays for vectorized lookups
    if settl_pos:
        settl_xy = np.array(settl_pos)  # (n_settl, 2) with (x, y)
        settl_xs_arr = settl_xy[:, 0]
        settl_ys_arr = settl_xy[:, 1]
    if port_pos:
        port_xy = np.array(port_pos)
        port_xs_arr = port_xy[:, 0]
        port_ys_arr = port_xy[:, 1]

    # Precompute component size/settl arrays for fast lookup
    comp_size_arr = np.zeros((h, w), dtype=np.int32)
    comp_settl_arr = np.zeros((h, w), dtype=np.int32)
    for comp_id, size in comp_sizes.items():
        mask = labeled == comp_id
        comp_size_arr[mask] = size
        comp_settl_arr[mask] = comp_settl.get(comp_id, 0)

    # Edge distance
    ys = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]
    dist_to_edge_arr = np.minimum(np.minimum(ys, h - 1 - ys), np.minimum(xs, w - 1 - xs))

    n_settl = len(settl_pos)
    features = []
    coords = []
    for y in range(h):
        for x in range(w):
            code = initial_grid[y][x]
            if code in {10, 5}:
                continue

            # Settlement distances (vectorized over settlements, loop over cells)
            if settl_pos:
                dists_all = np.abs(y - settl_ys_arr) + np.abs(x - settl_xs_arr)
                dist = int(dists_all.min())
                sorted_d = np.sort(dists_all)
                dist2 = int(sorted_d[1]) if len(sorted_d) >= 2 else 99
                nearby_settl = int((dists_all <= 4).sum())
                settlements_r3 = int((dists_all <= 3).sum())
                settl_r12 = int(((dists_all >= 1) & (dists_all <= 2)).sum())
                settl_r57 = int(((dists_all >= 5) & (dists_all <= 7)).sum())
            else:
                dist = dist2 = 99
                nearby_settl = settlements_r3 = settl_r12 = settl_r57 = 0

            if port_pos:
                dist_port = int((np.abs(y - port_ys_arr) + np.abs(x - port_xs_arr)).min())
            else:
                dist_port = 99

            adj_ocean = int(adj_ocean_arr[y, x])
            adj_forest = int(adj_forest_arr[y, x])
            adj_settl = int(adj_settl_arr[y, x])
            adj_mountain = int(adj_mountain_arr[y, x])
            bfs_d = int(bfs_dist[y, x])
            land_r3 = float(land_ratio_r3[y, x])

            # Terrain diversity in r3
            terrain_types = set()
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if abs(dy) + abs(dx) > 3:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        terrain_types.add(initial_grid[ny][nx])

            features.append([
                code, dist, adj_ocean, adj_forest, adj_settl, adj_mountain,
                int(code == 11), int(code == 4), int(code == 1), int(code == 2),
                int(adj_ocean >= 2),
                dist2, nearby_settl, n_settl,
                int(dist_ocean_arr[y, x]), int(dist_mountain_arr[y, x]), int(forests_r2_arr[y, x]),
                y, x,
                settlements_r3, settl_r12, settl_r57,
                dist_port, int(comp_size_arr[y, x]), int(comp_settl_arr[y, x]),
                int(dist_to_edge_arr[y, x]), bfs_d, int(passable_r2[y, x]), land_r3,
                land_r3 / (1 + bfs_d),
                len(terrain_types),
            ])
            coords.append((y, x))

    return np.array(features) if features else np.empty((0, 31)), coords


def load_gbt_models() -> Optional[list]:
    """Load pre-trained GBT models from pickle."""
    global _gbt_models
    if _gbt_models is not None:
        return _gbt_models

    path = os.path.join(os.path.dirname(__file__), "data", "gbt_models.pkl")
    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        _gbt_models = pickle.load(f)
    return _gbt_models


def compute_obs_stats(observations: list[dict]) -> np.ndarray:
    """Compute 22 enhanced round-level stats from observations.

    Features: base 7 (avg pop/food/wealth/defense, alive_rate, factions, port_rate)
    + 15 new (min/max/std of pop/food/defense, faction norm, grid settl/ruin rate,
      food deficit, dead rate, wealth stats).
    """
    pops, foods, wealths, defenses = [], [], [], []
    factions = set()
    alive_count, total_count, port_count = 0, 0, 0
    grid_settl_count, grid_ruin_count, grid_total = 0, 0, 0
    for obs in observations:
        for s in obs.get("settlements", []):
            total_count += 1
            if s.get("alive", True):
                alive_count += 1
                pops.append(s.get("population", 0))
                foods.append(s.get("food", 0))
                wealths.append(s.get("wealth", 0))
                defenses.append(s.get("defense", 0))
                factions.add(s.get("owner_id", -1))
                if s.get("has_port"):
                    port_count += 1
        for row in obs.get("grid", []):
            for code in row:
                if code not in {10, 5}:
                    grid_total += 1
                    if code in {1, 2}:
                        grid_settl_count += 1
                    elif code == 3:
                        grid_ruin_count += 1
    n_queries = max(len(observations), 1)
    base = np.array([
        np.mean(pops) if pops else 0.0,
        np.mean(foods) if foods else 0.0,
        np.mean(wealths) if wealths else 0.0,
        np.mean(defenses) if defenses else 0.0,
        alive_count / max(total_count, 1),
        len(factions) / max(n_queries, 1),
        port_count / max(alive_count, 1),
    ])
    min_food = float(np.min(foods)) if foods else 0.0
    max_pop = float(np.max(pops)) if pops else 0.0
    food_std = float(np.std(foods)) if foods else 0.0
    min_defense = float(np.min(defenses)) if defenses else 0.0
    defense_std = float(np.std(defenses)) if defenses else 0.0
    n_factions_norm = len(factions) / 10.0
    obs_settl_rate = grid_settl_count / max(grid_total, 1)
    obs_ruin_rate = grid_ruin_count / max(grid_total, 1)
    avg_food = float(np.mean(foods)) if foods else 0.0
    avg_pop = float(np.mean(pops)) if pops else 0.0
    food_deficit = avg_food - avg_pop
    pop_std = float(np.std(pops)) if pops else 0.0
    max_food = float(np.max(foods)) if foods else 0.0
    min_pop = float(np.min(pops)) if pops else 0.0
    max_defense = float(np.max(defenses)) if defenses else 0.0
    wealth_mean = float(np.mean(wealths)) if wealths else 0.0
    wealth_std = float(np.std(wealths)) if wealths else 0.0
    dead_rate = (total_count - alive_count) / max(total_count, 1)
    return np.concatenate([base, [min_food, max_pop, food_std, min_defense, defense_std,
                                   n_factions_norm, obs_settl_rate, obs_ruin_rate, food_deficit,
                                   pop_std, max_food, min_pop, max_defense, wealth_mean,
                                   wealth_std, dead_rate]])


def compute_cell_obs_features(observations: list[dict], h: int, w: int) -> np.ndarray:
    """Compute per-cell observation features (23 features per cell).

    For each cell, computes settlement/empty/ruin rates from viewport observations,
    neighbor settlement density, nearest observed settlement stats, and spatial aggregates.
    """
    cell_counts = np.zeros((h, w), dtype=np.int32)
    cell_settl = np.zeros((h, w), dtype=np.int32)
    cell_empty = np.zeros((h, w), dtype=np.int32)
    cell_ruin = np.zeros((h, w), dtype=np.int32)
    obs_settl_list = []
    for obs in (observations or []):
        vp = obs.get("viewport", {})
        vy, vx = vp.get("y", 0), vp.get("x", 0)
        obs_grid = obs.get("grid", [])
        for dy in range(len(obs_grid)):
            row = obs_grid[dy]
            for dx in range(len(row)):
                y2, x2 = vy + dy, vx + dx
                if 0 <= y2 < h and 0 <= x2 < w:
                    cell_counts[y2, x2] += 1
                    code = row[dx]
                    if code in {1, 2}:
                        cell_settl[y2, x2] += 1
                    elif code in {0, 11}:
                        cell_empty[y2, x2] += 1
                    elif code == 3:
                        cell_ruin[y2, x2] += 1
        for s in obs.get("settlements", []):
            if s.get("alive", True):
                obs_settl_list.append((s.get("y", 0), s.get("x", 0),
                                       s.get("population", 0), s.get("food", 0),
                                       s.get("wealth", 0), s.get("defense", 0)))
    total_obs = max(len(observations or []), 1)
    settl_rate = np.zeros((h, w), dtype=np.float32)
    ruin_rate = np.zeros((h, w), dtype=np.float32)
    result = np.zeros((h, w, 23), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            n = cell_counts[y, x]
            if n > 0:
                sr = cell_settl[y, x] / n
                rr = cell_ruin[y, x] / n
                settl_rate[y, x] = sr
                ruin_rate[y, x] = rr
                result[y, x, 0] = sr
                result[y, x, 1] = cell_empty[y, x] / n
                result[y, x, 2] = rr
                result[y, x, 3] = n / total_obs
    if obs_settl_list:
        settl_ys = np.array([s[0] for s in obs_settl_list], dtype=np.float32)
        settl_xs = np.array([s[1] for s in obs_settl_list], dtype=np.float32)
        settl_pops = np.array([s[2] for s in obs_settl_list], dtype=np.float32)
        settl_foods = np.array([s[3] for s in obs_settl_list], dtype=np.float32)
        settl_wealths = np.array([s[4] for s in obs_settl_list], dtype=np.float32)
        settl_defenses = np.array([s[5] for s in obs_settl_list], dtype=np.float32)
        pop_max = max(float(np.max(settl_pops)), 1.0)
        food_max = max(float(np.max(settl_foods)), 1.0)
        wealth_max = max(float(np.max(settl_wealths)), 1.0)
        defense_max = max(float(np.max(settl_defenses)), 1.0)
        for y in range(h):
            for x in range(w):
                dists = np.abs(settl_ys - y) + np.abs(settl_xs - x)
                nearest_idx = int(np.argmin(dists))
                result[y, x, 7] = settl_pops[nearest_idx] / pop_max
                result[y, x, 8] = settl_foods[nearest_idx] / food_max
                result[y, x, 9] = float(dists[nearest_idx]) / max(h + w, 1)
                result[y, x, 19] = settl_wealths[nearest_idx] / wealth_max
                result[y, x, 20] = settl_defenses[nearest_idx] / defense_max
    for y in range(h):
        for x in range(w):
            nbr_rates, nbr_count, max_r1 = [], 0, 0.0
            sum_ruin, sum_settl_r3, sum_ruin_r3 = 0.0, 0.0, 0.0
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if dy == 0 and dx == 0:
                        continue
                    md = abs(dy) + abs(dx)
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue
                    if md <= 2:
                        nbr_rates.append(settl_rate[ny, nx])
                        if settl_rate[ny, nx] > 0.1:
                            nbr_count += 1
                        sum_ruin += ruin_rate[ny, nx]
                    if md == 1 and settl_rate[ny, nx] > max_r1:
                        max_r1 = settl_rate[ny, nx]
                    if md <= 3:
                        sum_settl_r3 += settl_rate[ny, nx]
                        sum_ruin_r3 += ruin_rate[ny, nx]
            result[y, x, 4] = float(np.mean(nbr_rates)) if nbr_rates else 0.0
            result[y, x, 5] = float(nbr_count)
            result[y, x, 6] = max_r1
            result[y, x, 10] = float(sum(nbr_rates))
            result[y, x, 11] = sum_ruin
            result[y, x, 12] = result[y, x, 0] + result[y, x, 2]
            result[y, x, 21] = sum_settl_r3
            result[y, x, 22] = sum_ruin_r3
    log_settl = np.zeros((h, w), dtype=np.float32)
    log_ruin = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            ls = float(np.log1p(cell_settl[y, x]))
            lr = float(np.log1p(cell_ruin[y, x]))
            log_settl[y, x] = ls
            log_ruin[y, x] = lr
            result[y, x, 13] = ls
            result[y, x, 14] = float(np.log1p(cell_counts[y, x]))
            result[y, x, 15] = lr
            result[y, x, 16] = float(np.log1p(cell_settl[y, x] + cell_ruin[y, x]))
    for y in range(h):
        for x in range(w):
            s_ls, s_lr = 0.0, 0.0
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if (dy == 0 and dx == 0) or abs(dy) + abs(dx) > 2:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        s_ls += log_settl[ny, nx]
                        s_lr += log_ruin[ny, nx]
            result[y, x, 17] = s_ls
            result[y, x, 18] = s_lr
    return result


def gbt_predict(
    initial_grid: list[list[int]],
    settlements: list,
    obs_stats: Optional[np.ndarray] = None,
    cell_obs: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Generate predictions using pre-trained terrain-specific GBT ensembles.

    Features: 30 cell + 22 obs stats + 23 per-cell obs = 75 total.
    Returns H×W×6 tensor, or None if models not available.
    """
    models_dict = load_gbt_models()
    if models_dict is None:
        return None

    h = len(initial_grid)
    w = len(initial_grid[0]) if h > 0 else 0
    features, coords = _extract_cell_features(initial_grid, settlements)

    if len(features) == 0:
        return None

    # Append round-level observation stats (22 features)
    if obs_stats is not None:
        features = np.hstack([features, np.tile(obs_stats, (len(features), 1))])

    # Append per-cell observation features (23 features)
    if cell_obs is not None:
        cell_feats = np.array([cell_obs[y, x] for y, x in coords])
        features = np.hstack([features, cell_feats])

    tensor = np.zeros((h, w, NUM_CLASSES))
    # Static cells
    for y in range(h):
        for x in range(w):
            if initial_grid[y][x] == 10:
                tensor[y, x] = [1, 0, 0, 0, 0, 0]
            elif initial_grid[y][x] == 5:
                tensor[y, x] = [0, 0, 0, 0, 0, 1]

    # Terrain-specific predictions
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


# Empirical transitions pooled from R1-R9 GT (40 maps)
EMPIRICAL_TRANSITIONS = {
    1:  [0.4372, 0.3224, 0.0042, 0.0285, 0.2077, 0.0000],  # Settlement (n=1805)
    2:  [0.4699, 0.0939, 0.1827, 0.0234, 0.2301, 0.0000],  # Port (n=79)
    4:  [0.0893, 0.1447, 0.0097, 0.0151, 0.7412, 0.0000],  # Forest (n=13592)
    5:  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],  # Mountain
    10: [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # Ocean
    11: [0.7971, 0.1378, 0.0103, 0.0145, 0.0403, 0.0000],  # Plains (n=38934)
}

# Distance tables calibrated from R1-R9 GT (40 maps)
PLAINS_INLAND_BY_DISTANCE = {
    1:  [0.698, 0.215, 0.007, 0.020, 0.060, 0.000],  # n=5261
    2:  [0.733, 0.185, 0.010, 0.019, 0.054, 0.000],  # n=8474
    3:  [0.786, 0.146, 0.009, 0.016, 0.043, 0.000],  # n=8549
    4:  [0.822, 0.116, 0.010, 0.013, 0.039, 0.000],  # n=6454
    5:  [0.873, 0.086, 0.008, 0.010, 0.023, 0.000],  # n=4031
    6:  [0.902, 0.067, 0.009, 0.008, 0.014, 0.000],  # n=2281
    7:  [0.927, 0.049, 0.008, 0.006, 0.010, 0.000],  # n=1210
    8:  [0.949, 0.036, 0.006, 0.003, 0.005, 0.000],  # n=645
    99: [0.977, 0.017, 0.003, 0.001, 0.002, 0.000],  # n=663
}

PLAINS_COASTAL_BY_DISTANCE = {
    1:  [0.716, 0.099, 0.105, 0.018, 0.061, 0.000],  # n=70
    2:  [0.773, 0.078, 0.089, 0.015, 0.045, 0.000],  # n=206
    3:  [0.812, 0.071, 0.064, 0.013, 0.039, 0.000],  # n=272
    4:  [0.858, 0.049, 0.048, 0.010, 0.034, 0.000],  # n=243
    5:  [0.878, 0.050, 0.038, 0.008, 0.025, 0.000],  # n=207
    6:  [0.915, 0.038, 0.028, 0.007, 0.013, 0.000],  # n=140
    7:  [0.925, 0.033, 0.023, 0.006, 0.013, 0.000],  # n=89
    8:  [0.947, 0.030, 0.015, 0.003, 0.005, 0.000],  # n=56
    99: [0.979, 0.013, 0.005, 0.002, 0.002, 0.000],  # n=76
}

FOREST_INLAND_BY_DISTANCE = {
    1:  [0.135, 0.225, 0.007, 0.021, 0.611, 0.000],  # n=1855
    2:  [0.117, 0.194, 0.010, 0.019, 0.659, 0.000],  # n=3084
    3:  [0.093, 0.150, 0.008, 0.016, 0.732, 0.000],  # n=2960
    4:  [0.084, 0.120, 0.008, 0.014, 0.774, 0.000],  # n=2254
    5:  [0.052, 0.090, 0.008, 0.010, 0.839, 0.000],  # n=1364
    6:  [0.030, 0.070, 0.007, 0.008, 0.885, 0.000],  # n=744
    7:  [0.022, 0.053, 0.009, 0.005, 0.911, 0.000],  # n=459
    8:  [0.016, 0.040, 0.005, 0.004, 0.935, 0.000],  # n=226
    99: [0.004, 0.017, 0.002, 0.002, 0.975, 0.000],  # n=236
}

FOREST_COASTAL_BY_DISTANCE = {
    1:  [0.130, 0.122, 0.139, 0.019, 0.590, 0.000],  # n=35
    2:  [0.105, 0.079, 0.095, 0.015, 0.705, 0.000],  # n=53
    3:  [0.076, 0.068, 0.061, 0.013, 0.782, 0.000],  # n=83
    4:  [0.073, 0.045, 0.050, 0.010, 0.822, 0.000],  # n=80
    5:  [0.066, 0.045, 0.037, 0.009, 0.844, 0.000],  # n=54
    6:  [0.025, 0.026, 0.018, 0.006, 0.925, 0.000],  # n=34
    7:  [0.020, 0.020, 0.016, 0.005, 0.939, 0.000],  # n=23
    8:  [0.011, 0.019, 0.012, 0.003, 0.955, 0.000],  # n=17
    99: [0.006, 0.016, 0.005, 0.002, 0.971, 0.000],  # n=27
}


# ──────────────────────────────────────────────────────────────
# Layer 1: Static prediction
# ──────────────────────────────────────────────────────────────

def build_static_prediction(initial_grid: list[list[int]]) -> np.ndarray:
    """Build prediction tensor using empirical transition probabilities.

    Uses coastal/inland split with distance-based priors for Plains and Forests.
    Key insight: only coastal cells can become ports; inland P(port) = 0.
    """
    h = len(initial_grid)
    w = len(initial_grid[0]) if h > 0 else 0
    tensor = np.full((h, w, NUM_CLASSES), 1.0 / NUM_CLASSES)

    # Pre-compute settlement positions for distance calculation
    settlement_positions = []
    for y in range(h):
        for x in range(w):
            if initial_grid[y][x] in {1, 2}:
                settlement_positions.append((y, x))

    for y in range(h):
        for x in range(w):
            code = initial_grid[y][x]

            if code in {11, 0}:
                # Plains/Empty — use coastal/inland + distance-based transitions
                if settlement_positions:
                    dist = min(
                        abs(y - sy) + abs(x - sx)
                        for sy, sx in settlement_positions
                    )
                else:
                    dist = 99

                # Key insight: cells with only 1 ocean neighbor behave like
                # inland cells (high P(settl), ~zero P(port)). True port
                # formation requires 2+ adjacent ocean tiles.
                adj_ocean = _count_adjacent_ocean(initial_grid, y, x)
                exposed_coastal = adj_ocean >= 2
                table = PLAINS_COASTAL_BY_DISTANCE if exposed_coastal else PLAINS_INLAND_BY_DISTANCE

                for max_d in sorted(table.keys()):
                    if dist <= max_d:
                        tensor[y, x] = table[max_d]
                        break

            elif code == 4:
                # Forest — use coastal/inland + distance-based transitions
                if settlement_positions:
                    dist = min(
                        abs(y - sy) + abs(x - sx)
                        for sy, sx in settlement_positions
                    )
                else:
                    dist = 99

                adj_ocean = _count_adjacent_ocean(initial_grid, y, x)
                exposed_coastal = adj_ocean >= 2
                table = FOREST_COASTAL_BY_DISTANCE if exposed_coastal else FOREST_INLAND_BY_DISTANCE

                for max_d in sorted(table.keys()):
                    if dist <= max_d:
                        tensor[y, x] = table[max_d]
                        break

            elif code in EMPIRICAL_TRANSITIONS:
                tensor[y, x] = EMPIRICAL_TRANSITIONS[code]
            elif code == 3:
                tensor[y, x] = [0.15, 0.15, 0.05, 0.25, 0.35, 0.05]

    return tensor


# ──────────────────────────────────────────────────────────────
# Layer 2: Update with observations
# ──────────────────────────────────────────────────────────────

def update_with_observations(
    tensor: np.ndarray,
    observations: list[dict],
    seed_index: int,
) -> np.ndarray:
    """Update prediction tensor with observed simulation results.

    For each cell observed in a viewport, count how often each terrain
    class appeared across multiple stochastic runs. The more observations,
    the more reliable the frequency estimate.
    """
    # Collect per-cell observation counts
    h, w, _ = tensor.shape
    counts = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)
    obs_count = np.zeros((h, w), dtype=np.int32)

    for obs in observations:
        if obs.get("seed_index") != seed_index:
            continue

        vp = obs["viewport"]
        grid = obs["grid"]
        vy, vx = vp["y"], vp["x"]

        for gy, row in enumerate(grid):
            for gx, code in enumerate(row):
                abs_y = vy + gy
                abs_x = vx + gx
                if 0 <= abs_y < h and 0 <= abs_x < w:
                    cls = TERRAIN_TO_CLASS.get(code, 0)
                    counts[abs_y, abs_x, cls] += 1
                    obs_count[abs_y, abs_x] += 1

    # Update tensor where we have observations
    # Bayesian blend: with few observations, the prior (from Layer 1) should
    # dominate. With many observations, the frequency should dominate.
    # We use a pseudo-count approach: treat the prior as N_PRIOR virtual observations.
    N_PRIOR = 20  # prior strength — equivalent to 20 virtual observations
    observed_mask = obs_count > 0
    for y in range(h):
        for x in range(w):
            if observed_mask[y, x]:
                n = obs_count[y, x]
                freq = counts[y, x] / n
                prior = tensor[y, x]  # from Layer 1
                # Weighted average: more observations → more weight on freq
                weight = n / (n + N_PRIOR)
                tensor[y, x] = weight * freq + (1 - weight) * prior

    return tensor


# ──────────────────────────────────────────────────────────────
# Layer 3: Fill unobserved dynamic cells with informed priors
# ──────────────────────────────────────────────────────────────

def _is_coastal(grid: list[list[int]], y: int, x: int) -> bool:
    """Check if a cell is adjacent to ocean."""
    h, w = len(grid), len(grid[0])
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and grid[ny][nx] == 10:
                return True
    return False


def _count_adjacent_ocean(grid: list[list[int]], y: int, x: int) -> int:
    """Count ocean cells adjacent to (y, x)."""
    h, w = len(grid), len(grid[0])
    count = 0
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and grid[ny][nx] == 10:
                count += 1
    return count


def _count_adjacent_forests(grid: list[list[int]], y: int, x: int) -> int:
    """Count forest cells adjacent to (y, x)."""
    h, w = len(grid), len(grid[0])
    count = 0
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and grid[ny][nx] == 4:
                count += 1
    return count


def _distance_to_nearest_settlement(
    grid: list[list[int]],
    y: int,
    x: int,
    settlements: list[dict],
) -> float:
    """Manhattan distance to nearest settlement."""
    if not settlements:
        return float("inf")
    return min(
        abs(y - s.get("y", 0)) + abs(x - s.get("x", 0))
        for s in settlements
    )


def fill_unobserved_dynamic(
    tensor: np.ndarray,
    initial_grid: list[list[int]],
    settlements: list[dict],
    observations: list[dict],
    seed_index: int,
) -> np.ndarray:
    """Fill unobserved dynamic cells with heuristic priors.

    Uses initial state analysis + simulation mechanics:
    - Settlements near forests get food → higher survival
    - Coastal settlements may become ports
    - Isolated settlements tend to become ruins
    - Empty land near settlements may get settled (expansion)
    """
    h, w, _ = tensor.shape

    # Build mask of observed cells
    obs_mask = np.zeros((h, w), dtype=bool)
    for obs in observations:
        if obs.get("seed_index") != seed_index:
            continue
        vp = obs["viewport"]
        vy, vx, vh, vw = vp["y"], vp["x"], vp["h"], vp["w"]
        obs_mask[vy:vy + vh, vx:vx + vw] = True

    for y in range(h):
        for x in range(w):
            code = initial_grid[y][x]

            # Skip already-observed cells
            if obs_mask[y, x]:
                continue

            # Skip truly static cells (ocean, mountain)
            if code in {10, 5}:
                continue

            adj_ocean = _count_adjacent_ocean(initial_grid, y, x)
            exposed_coastal = adj_ocean >= 2
            adj_forests = _count_adjacent_forests(initial_grid, y, x)
            dist = _distance_to_nearest_settlement(initial_grid, y, x, settlements)

            if code == 1:  # Settlement — R1+R2 pooled
                if exposed_coastal:
                    # Coastal settlements can become ports
                    tensor[y, x] = [0.355, 0.140, 0.328, 0.025, 0.151, 0.000]
                elif adj_forests == 0:
                    tensor[y, x] = [0.415, 0.365, 0.000, 0.033, 0.187, 0.000]
                elif adj_forests == 1:
                    tensor[y, x] = [0.386, 0.407, 0.000, 0.031, 0.176, 0.000]
                elif adj_forests == 2:
                    tensor[y, x] = [0.372, 0.422, 0.000, 0.033, 0.173, 0.000]
                else:  # 3+
                    tensor[y, x] = [0.362, 0.436, 0.000, 0.034, 0.168, 0.000]

            elif code == 2:  # Port — limited samples, use empirical
                tensor[y, x] = [0.320, 0.130, 0.310, 0.020, 0.220, 0.000]

            elif code == 3:  # Ruin — no ground truth data, keep heuristic
                if dist <= 3:
                    tensor[y, x] = [0.15, 0.20, 0.05, 0.25, 0.30, 0.05]
                else:
                    tensor[y, x] = [0.10, 0.05, 0.02, 0.25, 0.53, 0.05]

            elif code == 4:  # Forest — distance + exposed coastal from GT tables
                table = FOREST_COASTAL_BY_DISTANCE if exposed_coastal else FOREST_INLAND_BY_DISTANCE
                for max_d in sorted(table.keys()):
                    if dist <= max_d:
                        base = np.array(table[max_d], dtype=np.float64)
                        break
                else:
                    base = np.array(table[99], dtype=np.float64)

                tensor[y, x] = base

            elif code in {0, 11}:  # Plains — distance + exposed coastal from GT tables
                table = PLAINS_COASTAL_BY_DISTANCE if exposed_coastal else PLAINS_INLAND_BY_DISTANCE
                for max_d in sorted(table.keys()):
                    if dist <= max_d:
                        base = np.array(table[max_d], dtype=np.float64)
                        break
                else:
                    base = np.array(table[99], dtype=np.float64)

                # adj_settlement boost removed — XGBoost captures this via features

                tensor[y, x] = base

    return tensor


# ──────────────────────────────────────────────────────────────
# Layer 4: Cross-seed transfer
# ──────────────────────────────────────────────────────────────

def apply_cross_seed_transfer(
    tensor: np.ndarray,
    initial_grid: list[list[int]],
    all_observations: list[dict],
    seed_index: int,
    all_initial_grids: list[list[list[int]]],
    weight: float = 0.3,
) -> np.ndarray:
    """Pool observations from other seeds for cells with similar initial terrain.

    All seeds share the same hidden parameters. If a cell has the same initial
    terrain code and similar neighbors across seeds, observations from other seeds
    can inform our prediction (with lower weight).
    """
    h, w, _ = tensor.shape

    # Build observation frequency from OTHER seeds
    other_counts = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)
    other_obs_count = np.zeros((h, w), dtype=np.int32)

    for obs in all_observations:
        if obs.get("seed_index") == seed_index:
            continue  # Skip same seed

        other_seed = obs["seed_index"]
        # Only transfer if the initial terrain matches
        other_grid = all_initial_grids[other_seed] if other_seed < len(all_initial_grids) else None
        if other_grid is None:
            continue

        vp = obs["viewport"]
        grid = obs["grid"]
        vy, vx = vp["y"], vp["x"]

        for gy, row in enumerate(grid):
            for gx, code in enumerate(row):
                abs_y = vy + gy
                abs_x = vx + gx
                if 0 <= abs_y < h and 0 <= abs_x < w:
                    # Only transfer if initial terrain matches
                    if other_grid[abs_y][abs_x] == initial_grid[abs_y][abs_x]:
                        cls = TERRAIN_TO_CLASS.get(code, 0)
                        other_counts[abs_y, abs_x, cls] += 1
                        other_obs_count[abs_y, abs_x] += 1

    # Blend other-seed observations into tensor
    for y in range(h):
        for x in range(w):
            if other_obs_count[y, x] > 0 and initial_grid[y][x] not in STATIC_TERRAIN_CODES:
                other_freq = other_counts[y, x] / other_obs_count[y, x]
                tensor[y, x] = (1 - weight) * tensor[y, x] + weight * other_freq

    return tensor


# ──────────────────────────────────────────────────────────────
# Layer 5: Calibration from past rounds
# ──────────────────────────────────────────────────────────────

def load_calibration() -> Optional[dict]:
    """Load calibration data from past rounds."""
    path = os.path.join(os.path.dirname(__file__), "data", "calibration.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def apply_calibration(
    tensor: np.ndarray,
    initial_grid: list[list[int]],
    calibration: Optional[dict],
) -> np.ndarray:
    """Adjust predictions using calibration data learned from past rounds.

    Calibration data contains learned transition probabilities:
    e.g., P(settlement survives | 3+ adjacent forests) = 0.72
    """
    if calibration is None:
        return tensor

    h, w, _ = tensor.shape

    # Apply calibrated priors for specific terrain/context combinations
    priors = calibration.get("context_priors", {})

    for y in range(h):
        for x in range(w):
            code = initial_grid[y][x]
            if code in STATIC_TERRAIN_CODES:
                continue

            key = str(code)
            coastal = _is_coastal(initial_grid, y, x)
            adj_forests = _count_adjacent_forests(initial_grid, y, x)

            # Build context key
            context = f"{key}_coastal{int(coastal)}_forests{min(adj_forests, 3)}"
            if context in priors:
                calibrated = np.array(priors[context], dtype=np.float64)
                # Blend calibration with current prediction (calibration has priority)
                # Tuned: 0.16 is optimal (grid searched 0.05-0.20)
                blend_weight = 0.16
                tensor[y, x] = (1 - blend_weight) * tensor[y, x] + blend_weight * calibrated

    return tensor


# ──────────────────────────────────────────────────────────────
# Ensemble support
# ──────────────────────────────────────────────────────────────

def ensemble_predictions(
    predictions: list[np.ndarray],
    weights: Optional[list[float]] = None,
) -> np.ndarray:
    """Average multiple prediction tensors, optionally with weights.

    Useful for combining different strategies:
    - Conservative (high-floor priors) vs aggressive (sharp distributions)
    - Different viewport allocations
    - With/without cross-seed transfer
    """
    if weights is None:
        weights = [1.0 / len(predictions)] * len(predictions)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    result = np.zeros_like(predictions[0])
    for pred, w in zip(predictions, weights):
        result += w * pred
    return result


def simulator_predict(initial_grid, settlements, observations=None, n_sims=300):
    """Generate predictions using Monte Carlo simulator only."""
    if observations:
        params = fit_hidden_params(initial_grid, settlements, observations)
    else:
        params = {"expansion_rate": 0.087, "winter_severity": 0.090, "raid_intensity": 0.03}
    return simulate_monte_carlo(
        initial_grid, settlements, n_sims=n_sims, **params
    )


# ──────────────────────────────────────────────────────────────
# Cross-seed empirical distance tables (built per-round from ALL observations)
# ──────────────────────────────────────────────────────────────

DIST_BUCKETS = [1, 2, 3, 4, 5, 6, 7, 8, 99]


def _dist_to_bucket(d: int) -> int:
    """Map a Manhattan distance to the nearest bucket."""
    for b in DIST_BUCKETS:
        if d <= b:
            return b
    return 99


def _is_coastal_4connected(grid: list[list[int]], y: int, x: int) -> bool:
    """Check if a cell has 2+ ocean neighbors (4-connected)."""
    h, w = len(grid), len(grid[0])
    count = 0
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and grid[ny][nx] == 10:
            count += 1
    return count >= 2


def build_round_empirical_tables(
    all_observations: list[dict],
    all_initial_grids: list[list[list[int]]],
    all_settlements: list[list[dict]] | None = None,
) -> dict:
    """Build empirical P(class) tables from pooled cross-seed observations.

    Groups observed cells by (initial_terrain, distance_bucket, coastal_status)
    and computes empirical class distributions. This adapts to the round's
    hidden parameters at the distance level.

    Args:
        all_observations: ALL observations across ALL seeds for this round
        all_initial_grids: initial grid for each seed
        all_settlements: settlements list for each seed (for distance computation)

    Returns:
        dict mapping (terrain_code, dist_bucket, coastal_bool) -> np.array of P(class)
    """
    if not all_observations or not all_initial_grids:
        return {}

    # Pre-compute settlement positions per seed
    seed_settl_pos = {}
    if all_settlements:
        for si, settls in enumerate(all_settlements):
            positions = []
            for s in settls:
                if isinstance(s, dict):
                    positions.append((s.get("x", 0), s.get("y", 0)))
                else:
                    positions.append((s.x, s.y))
            seed_settl_pos[si] = positions
    else:
        # Fall back to extracting settlements from initial grids
        for si, grid in enumerate(all_initial_grids):
            positions = []
            h, w = len(grid), len(grid[0])
            for y in range(h):
                for x in range(w):
                    if grid[y][x] in {1, 2}:
                        positions.append((x, y))
            seed_settl_pos[si] = positions

    # Accumulate counts per bucket
    buckets = {}  # (terrain_code, dist_bucket, coastal) -> counts[NUM_CLASSES]

    for obs in all_observations:
        seed_idx = obs.get("seed_index", 0)
        if seed_idx >= len(all_initial_grids):
            continue

        grid = all_initial_grids[seed_idx]
        h, w = len(grid), len(grid[0])
        settl_pos = seed_settl_pos.get(seed_idx, [])

        vp = obs.get("viewport", {})
        vy, vx = vp.get("y", 0), vp.get("x", 0)
        obs_grid = obs.get("grid", [])

        for dy in range(len(obs_grid)):
            for dx in range(len(obs_grid[0]) if obs_grid else 0):
                abs_y = vy + dy
                abs_x = vx + dx
                if not (0 <= abs_y < h and 0 <= abs_x < w):
                    continue

                initial_code = grid[abs_y][abs_x]
                # Skip static terrain (ocean=10, mountain=5)
                if initial_code in {10, 5}:
                    continue

                # Compute distance to nearest settlement
                if settl_pos:
                    dist = min(
                        abs(abs_y - sy) + abs(abs_x - sx)
                        for sx, sy in settl_pos
                    )
                else:
                    dist = 99
                dist_bucket = _dist_to_bucket(dist)

                # Coastal status (4-connected, 2+ ocean)
                coastal = _is_coastal_4connected(grid, abs_y, abs_x)

                # Observed terrain class
                observed_code = obs_grid[dy][dx]
                cls = TERRAIN_TO_CLASS.get(observed_code, 0)

                key = (initial_code, dist_bucket, coastal)
                if key not in buckets:
                    buckets[key] = np.zeros(NUM_CLASSES)
                buckets[key][cls] += 1

    # Convert counts to probabilities (require minimum 10 observations)
    result = {}
    for key, counts in buckets.items():
        total = counts.sum()
        if total >= 10:
            probs = counts / total
            probs = np.maximum(probs, PROB_FLOOR)
            probs /= probs.sum()
            result[key] = probs

    return result


# ──────────────────────────────────────────────────────────────
# Full prediction pipeline
# ──────────────────────────────────────────────────────────────

def build_prediction(
    initial_grid: list[list[int]],
    settlements: list[dict],
    observations: list[dict],
    seed_index: int,
    all_initial_grids: list[list[list[int]]],
    all_observations: list[dict],
    calibration: Optional[dict] = None,
    all_settlements: list[list[dict]] | None = None,
    round_empirical_tables: dict | None = None,
    sim_params: dict | None = None,
) -> np.ndarray:
    """Build a complete prediction tensor for one seed.

    Applies all prediction layers in sequence, then normalizes.
    """
    # Layer 1: Static prediction
    tensor = build_static_prediction(initial_grid)

    # Layer 2: Observation-based frequency update (disabled — adds noise with
    # only ~10 observations per seed; GT-calibrated priors outperform raw frequencies)
    # tensor = update_with_observations(tensor, observations, seed_index)

    # Layer 3: Apply context-specific priors to ALL dynamic cells.
    # With Layer 2 disabled, Layer 3's priors (adj_forests, coastal) are better
    # than Layer 1's distance-only tables for settlements. Pass empty observations
    # to bypass the obs_mask and fill everything.
    tensor = fill_unobserved_dynamic(
        tensor, initial_grid, settlements, [], seed_index,
    )

    # Layer 4: Cross-seed transfer (disabled — hurts score by ~0.5 pts)
    # tensor = apply_cross_seed_transfer(
    #     tensor, initial_grid, all_observations, seed_index, all_initial_grids,
    # )

    # Layer 5: Calibration (disabled — GBT subsumes calibration's role)
    # tensor = apply_calibration(tensor, initial_grid, calibration)

    # Layer 6: GBT blend — captures feature interactions the tables miss
    # Enhanced obs stats (22 features) + per-cell obs features (23 features)
    obs_stats = compute_obs_stats(all_observations) if all_observations else None
    h_map, w_map = len(initial_grid), len(initial_grid[0]) if initial_grid else 0
    cell_obs = compute_cell_obs_features(all_observations, h_map, w_map) if all_observations else None
    gbt_pred = gbt_predict(initial_grid, settlements, obs_stats=obs_stats, cell_obs=cell_obs)
    if gbt_pred is not None:
        # Per-terrain blend weights (from params.py)
        h, w, _ = tensor.shape
        for y in range(h):
            for x in range(w):
                code = initial_grid[y][x]
                if code in {10, 5}:
                    continue
                if code in {11, 0}:
                    bw = BLEND_TERRAIN["plains"]
                elif code == 4:
                    bw = BLEND_TERRAIN["forest"]
                elif code in {1, 2}:
                    bw = BLEND_TERRAIN["settl"]
                else:
                    bw = GBT_BLEND_WEIGHT
                tensor[y, x] = (1 - bw) * tensor[y, x] + bw * gbt_pred[y, x]

    # Layer 6.5: Simulator ensemble — Monte Carlo simulation blended with
    # XGBoost+heuristic predictions. Only applied to dynamic cells.
    # Use refined params if provided, otherwise fall back to fit_hidden_params.
    # Weight increased to 0.15 when refined params are used.
    current_sim_weight = SIM_BLEND_WEIGHT
    if sim_params:
        current_sim_weight = max(current_sim_weight, 0.15)
        
    if current_sim_weight > 0:
        try:
            if sim_params is None:
                sim_params = fit_hidden_params(initial_grid, settlements, all_observations)
            
            sim_probs = simulate_monte_carlo(
                initial_grid, settlements, n_sims=300, **sim_params
            )
            grid_arr = np.array(initial_grid)
            dynamic_mask = ~np.isin(grid_arr, [10, 5])
            tensor[dynamic_mask] = (
                (1 - current_sim_weight) * tensor[dynamic_mask]
                + current_sim_weight * sim_probs[dynamic_mask]
            )
        except Exception:
            pass  # gracefully fall back to no blending if simulator fails

    # Layer 6.7: Round-specific empirical distance tables.
    # Pool ALL observations across ALL seeds, group by (initial_terrain, distance_bucket, coastal),
    # and compute empirical P(class). This adapts to THIS round's hidden parameters at the distance level.
    if round_empirical_tables is None and all_observations:
        round_empirical_tables = build_round_empirical_tables(
            all_observations, all_initial_grids,
            all_settlements=all_settlements,
        )

    if round_empirical_tables:
        h, w, _ = tensor.shape
        # Round-type-aware empirical blend (from params.py)
        round_type = classify_round_type(obs_stats)
        EMPIRICAL_BLEND_WEIGHT = EMP_BLEND_EXPANSION if round_type == "expansion" else EMP_BLEND_NORMAL
        MIN_BUCKET_OBS = 20  # minimum obs in bucket to trust empirical

        # Pre-compute settlement positions for this seed
        _settl_pos = []
        for s in settlements:
            if isinstance(s, dict):
                _settl_pos.append((s.get("x", 0), s.get("y", 0)))
            else:
                _settl_pos.append((s.x, s.y))

        for y in range(h):
            for x in range(w):
                code = initial_grid[y][x]
                if code in {10, 5}:
                    continue

                # Compute distance and bucket
                if _settl_pos:
                    dist = min(abs(y - sy) + abs(x - sx) for sx, sy in _settl_pos)
                else:
                    dist = 99
                dist_bucket = _dist_to_bucket(dist)
                coastal = _is_coastal_4connected(initial_grid, y, x)

                key = (code, dist_bucket, coastal)
                if key in round_empirical_tables:
                    emp_probs = round_empirical_tables[key]
                    tensor[y, x] = (
                        (1 - EMPIRICAL_BLEND_WEIGHT) * tensor[y, x]
                        + EMPIRICAL_BLEND_WEIGHT * emp_probs
                    )

    # Layer 7: Observation-based ratio correction.
    # Use pooled observations from ALL seeds to estimate the hidden expansion
    # rate, then adjust predictions to match observed terrain frequencies.
    # This adapts the model to each round's unique hidden parameters.
    if all_observations:
        obs_cls = np.zeros(NUM_CLASSES)
        obs_total = 0
        for obs in all_observations:
            for row in obs.get("grid", []):
                for code in row:
                    if code not in {10, 5}:  # skip static
                        obs_cls[TERRAIN_TO_CLASS.get(code, 0)] += 1
                        obs_total += 1

        if obs_total > 100:  # need enough observations
            obs_freq = obs_cls / obs_total
            # Compute model's average prediction for dynamic cells (vectorized)
            h, w, _ = tensor.shape
            grid_arr = np.array(initial_grid)
            dynamic = ~np.isin(grid_arr, [10, 5])
            model_avg = tensor[dynamic].mean(axis=0)
            ratio = obs_freq / np.maximum(model_avg, 1e-6)
            # Round-type-adaptive L7 (synced from train.py LORO-validated)
            round_type = classify_round_type(obs_stats)
            rt_params = ROUND_TYPE_L7[round_type]
            strengths = rt_params["strengths"].copy()
            for c in range(NUM_CLASSES):
                if obs_cls[c] < L7_MIN_OBS[c]:
                    strengths[c] = 0.0
            adj = 1.0 + strengths * (ratio - 1.0)
            adj = np.clip(adj, rt_params["adj_min"], rt_params["adj_max"])
            # Apply to all dynamic cells at once
            tensor[dynamic] *= adj[np.newaxis, :]
            tensor[dynamic] = np.maximum(tensor[dynamic], PROB_FLOOR)
            tensor[dynamic] /= tensor[dynamic].sum(axis=1, keepdims=True)

    # Layer 8: Per-cell empirical correction from observation grids.
    # Each observation is a Monte Carlo sample of the final state.
    # For cells with multiple observations, use empirical frequencies
    # to correct the model prediction (Bayesian blend).
    if all_observations:
        h, w, _ = tensor.shape
        cell_counts = np.zeros((h, w), dtype=np.int32)
        cell_terrain = np.zeros((h, w, NUM_CLASSES))
        for obs in all_observations:
            vp = obs.get("viewport", {})
            vy = vp.get("y", 0)
            vx = vp.get("x", 0)
            obs_grid = obs.get("grid", [])
            for dy in range(len(obs_grid)):
                for dx in range(len(obs_grid[0]) if obs_grid else 0):
                    y, x = vy + dy, vx + dx
                    if 0 <= y < h and 0 <= x < w:
                        cell_counts[y, x] += 1
                        code = obs_grid[dy][dx]
                        cls = TERRAIN_TO_CLASS.get(code, 0)
                        cell_terrain[y, x, cls] += 1

        # Blend model prediction with empirical for cells with enough samples
        MIN_SAMPLES = 50  # effectively disabled until repeated viewports deployed
        EMP_WEIGHT_PER_SAMPLE = 0.03  # 3% weight per sample, capped at 30%
        MAX_EMP_WEIGHT = 0.30
        for y in range(h):
            for x in range(w):
                n = cell_counts[y, x]
                if n >= MIN_SAMPLES and initial_grid[y][x] not in {10, 5}:
                    emp = cell_terrain[y, x] / n
                    emp = np.maximum(emp, PROB_FLOOR)
                    emp /= emp.sum()
                    alpha = min(MAX_EMP_WEIGHT, n * EMP_WEIGHT_PER_SAMPLE)
                    tensor[y, x] = (1 - alpha) * tensor[y, x] + alpha * emp
                    tensor[y, x] = np.maximum(tensor[y, x], PROB_FLOOR)
                    tensor[y, x] /= tensor[y, x].sum()

    # Final normalization — CRITICAL: enforce floor + renormalize
    tensor = normalize_prediction(tensor)

    return tensor
