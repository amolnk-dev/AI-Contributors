"""Autoresearch-compatible training/evaluation script for Astar Island.

This is the MUTABLE file — the autoresearch agent edits this.
Runs offline 5-fold LORO evaluation using local GT data (~75s).

Usage:
    cd tasks/astar-island && python train.py > run.log 2>&1
    grep "^val_metric:" run.log

Contract:
    - TIME_BUDGET: ~75 seconds (5-fold LORO with XGBoost retraining)
    - Prints val_metric: <float> (higher is better, max 100)
    - Prints per-round breakdown for diagnostics
"""

import json
import os
import sys
import time
import numpy as np
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from scipy.ndimage import gaussian_filter
import xgboost as xgb
from model import (
    _extract_cell_features,
    build_static_prediction,
    fill_unobserved_dynamic,
    compute_obs_stats,
    compute_cell_obs_features,
    build_round_empirical_tables,
    _dist_to_bucket,
    _is_coastal_4connected,
)
from evaluate import compute_score
from utils import normalize_prediction, load_observations
from dtos import TERRAIN_TO_CLASS, NUM_CLASSES, PROB_FLOOR
from params import (
    BLEND_WEIGHT, BLEND_TERRAIN, SMOOTH_WEIGHT,
    L7_STRENGTHS, L7_MIN_OBS, L7_ADJ_MIN, L7_ADJ_MAX,
    ROUND_TYPE_THRESHOLDS, ROUND_TYPE_L7,
    SPATIAL_SMOOTH_SIGMA, SPATIAL_SMOOTH_BLEND,
    XGB_HPARAMS, ROUNDS, ROUND_WEIGHTS,
    EMP_BLEND_EXPANSION, EMP_BLEND_NORMAL,
    classify_round_type,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def spatial_smooth(tensor, grid, sigma=SPATIAL_SMOOTH_SIGMA, blend=SPATIAL_SMOOTH_BLEND):
    """Apply Gaussian spatial smoothing to settlement and forest channels.
    Only smooth dynamic cells; preserve static ocean/mountain.
    """
    if sigma <= 0 or blend <= 0:
        return tensor
    h, w, _ = tensor.shape
    grid_arr = np.array(grid)
    dynamic = ~np.isin(grid_arr, [10, 5])

    # Smooth settlement channel (class 1) and forest channel (class 4)
    for ch in [1, 4]:
        channel = tensor[:, :, ch].copy()
        # Mask out static cells before smoothing
        channel[~dynamic] = 0.0
        smoothed = gaussian_filter(channel, sigma=sigma)
        # Blend smoothed back into tensor for dynamic cells only
        tensor[:, :, ch][dynamic] = (
            (1 - blend) * tensor[:, :, ch][dynamic] + blend * smoothed[dynamic]
        )

    # Re-normalize dynamic cells
    tensor[dynamic] = np.maximum(tensor[dynamic], PROB_FLOOR)
    tensor[dynamic] /= tensor[dynamic].sum(axis=1, keepdims=True)
    return tensor


def load_round_data(round_num):
    with open(os.path.join(DATA_DIR, f"round{round_num}_initial.json")) as f:
        info = json.load(f)
    initial_states = info["initial_states"]
    gts = []
    for seed in range(5):
        gts.append(np.load(os.path.join(DATA_DIR, f"gt_r{round_num}_seed{seed}.npy")))
    return initial_states, gts


def train_gbt_models(train_rounds):
    """Train terrain-specific XGBoost on given rounds (76 features: 30 cell + 23 obs stats + 23 cell obs)."""
    X_data = {"plains": [], "forest": [], "settl": []}
    Y_data = {"plains": [], "forest": [], "settl": []}
    W_data = {"plains": [], "forest": [], "settl": []}

    for rnum in train_rounds:
        initial_states, gts = load_round_data(rnum)
        all_obs = load_observations(ROUNDS[rnum])
        obs_stats = compute_obs_stats(all_obs) if all_obs else np.zeros(23)
        rw = ROUND_WEIGHTS.get(rnum, 1.0)
        # Compute cell-level obs features (shared across seeds for this round)
        # Use seed 0 grid dimensions (all seeds same map size)
        g0 = initial_states[0]["grid"]
        h0, w0 = len(g0), len(g0[0]) if g0 else 0
        cell_obs = compute_cell_obs_features(all_obs, h0, w0)
        for seed in range(5):
            grid = initial_states[seed]["grid"]
            settlements = initial_states[seed]["settlements"]
            gt = gts[seed]
            feats, coords = _extract_cell_features(grid, settlements)
            # Append round-level obs stats (46 total so far)
            feats = np.hstack([feats, np.tile(obs_stats, (len(feats), 1))])
            # Append per-cell obs features (50 total)
            cell_feats = np.array([cell_obs[y, x] for y, x in coords])
            feats = np.hstack([feats, cell_feats])
            targets = np.array([gt[y, x] for y, x in coords])
            for i, (y, x) in enumerate(coords):
                code = grid[y][x]
                sample_w = rw
                if code in {11, 0}:
                    X_data["plains"].append(feats[i])
                    Y_data["plains"].append(targets[i])
                    W_data["plains"].append(sample_w)
                elif code == 4:
                    X_data["forest"].append(feats[i])
                    Y_data["forest"].append(targets[i])
                    W_data["forest"].append(sample_w)
                elif code in {1, 2}:
                    X_data["settl"].append(feats[i])
                    Y_data["settl"].append(targets[i])
                    W_data["settl"].append(sample_w)

    # Convert to arrays
    for tt in ["plains", "forest", "settl"]:
        X_data[tt] = np.array(X_data[tt]) if X_data[tt] else np.empty((0, 0))
        Y_data[tt] = np.array(Y_data[tt]) if Y_data[tt] else np.empty((0, 6))
        W_data[tt] = np.array(W_data[tt]) if W_data[tt] else np.empty(0)

    # Detect parallelism: use PARALLEL_XGB env var or default based on CPU count
    n_workers = int(os.environ.get("PARALLEL_XGB", "0"))
    if n_workers == 0:
        n_workers = 1 if os.cpu_count() <= 16 else 18

    if n_workers > 1:
        return _train_gbt_parallel(X_data, Y_data, W_data, n_workers)
    else:
        return _train_gbt_sequential(X_data, Y_data, W_data)


def _train_gbt_sequential(X_data, Y_data, W_data):
    """Train 18 XGBoost models sequentially (default for <=16 cores)."""
    models = {}
    for terrain_type in ["plains", "forest", "settl"]:
        X = X_data[terrain_type]
        Y = Y_data[terrain_type]
        W = W_data[terrain_type]
        if len(X) == 0:
            continue
        hp = XGB_HPARAMS[terrain_type]
        terrain_models = []
        for cls in range(6):
            m = xgb.XGBRegressor(
                n_estimators=hp["n_estimators"],
                max_depth=hp["max_depth"],
                learning_rate=hp["learning_rate"],
                reg_alpha=hp["reg_alpha"],
                reg_lambda=hp["reg_lambda"],
                subsample=hp["subsample"],
                colsample_bytree=hp["colsample_bytree"],
                min_child_weight=hp["min_child_weight"],
                random_state=42, verbosity=0,
            )
            m.fit(X, Y[:, cls], sample_weight=W)
            terrain_models.append(m)
        models[terrain_type] = terrain_models
    return models


def _fit_one_model(args):
    """Train a single XGBoost model (for parallel execution)."""
    X, y, w, hp = args
    m = xgb.XGBRegressor(
        n_estimators=hp["n_estimators"],
        max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"],
        reg_alpha=hp["reg_alpha"],
        reg_lambda=hp["reg_lambda"],
        subsample=hp["subsample"],
        colsample_bytree=hp["colsample_bytree"],
        min_child_weight=hp["min_child_weight"],
        random_state=42, verbosity=0, nthread=1,
    )
    m.fit(X, y, sample_weight=w)
    return m


def _train_gbt_parallel(X_data, Y_data, W_data, n_workers=18):
    """Train 18 XGBoost models in parallel (for high-core-count machines)."""
    from concurrent.futures import ProcessPoolExecutor

    tasks = []
    task_keys = []  # (terrain_type, cls)
    for terrain_type in ["plains", "forest", "settl"]:
        X = X_data[terrain_type]
        Y = Y_data[terrain_type]
        W = W_data[terrain_type]
        if len(X) == 0:
            continue
        hp = XGB_HPARAMS[terrain_type]
        for cls in range(6):
            tasks.append((X, Y[:, cls], W, hp))
            task_keys.append((terrain_type, cls))

    with ProcessPoolExecutor(max_workers=min(n_workers, len(tasks))) as pool:
        fitted = list(pool.map(_fit_one_model, tasks))

    models = {}
    for (tt, cls), model in zip(task_keys, fitted):
        if tt not in models:
            models[tt] = [None] * 6
        models[tt][cls] = model
    return models


def gbt_predict_with_models(models_dict, initial_grid, settlements, obs_stats=None, cell_obs=None):
    h = len(initial_grid)
    w = len(initial_grid[0]) if h > 0 else 0
    features, coords = _extract_cell_features(initial_grid, settlements)
    if len(features) == 0:
        return None
    # Append obs stats (always 23 features — matches model.py compute_obs_stats)
    if obs_stats is None:
        obs_stats = np.zeros(23)
    features = np.hstack([features, np.tile(obs_stats, (len(features), 1))])
    # Append per-cell obs features (always 23 features — matches model.py compute_cell_obs_features)
    if cell_obs is None:
        cell_obs = np.zeros((h, w, 24))
    cell_feats = np.array([cell_obs[y, x] for y, x in coords])
    features = np.hstack([features, cell_feats])

    # Build terrain type array for all coords
    grid_arr = np.array(initial_grid)
    tensor = np.zeros((h, w, NUM_CLASSES))
    tensor[grid_arr == 10] = [1, 0, 0, 0, 0, 0]
    tensor[grid_arr == 5] = [0, 0, 0, 0, 0, 1]

    # Batch predict per terrain type (18 predict calls instead of ~7200)
    coord_arr = np.array(coords)  # (n_cells, 2)
    codes = np.array([initial_grid[y][x] for y, x in coords])
    terrain_masks = {
        "plains": np.isin(codes, [11, 0]),
        "forest": codes == 4,
        "settl": np.isin(codes, [1, 2]),
    }

    for ttype, mask in terrain_masks.items():
        if not mask.any():
            continue
        models = models_dict.get(ttype)
        if models is None:
            continue
        batch_feats = features[mask]
        batch_coords = coord_arr[mask]
        preds = np.column_stack([models[cls].predict(batch_feats) for cls in range(NUM_CLASSES)])
        preds = np.maximum(preds, PROB_FLOOR)
        preds /= preds.sum(axis=1, keepdims=True)
        for j in range(len(batch_coords)):
            tensor[batch_coords[j, 0], batch_coords[j, 1]] = preds[j]

    return tensor


def evaluate_loro():
    """Run full LORO and return (avg, per_round_dict).

    Supports LORO_FOLDS env var for quick screening:
        LORO_FOLDS=7,13,16 python train.py  → only evaluate those 3 folds
    """
    all_rounds = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21]  # R12 excluded: 0 observations

    # Support quick screening: only evaluate specified folds
    loro_folds_env = os.environ.get("LORO_FOLDS", "")
    if loro_folds_env:
        test_rounds = [int(r) for r in loro_folds_env.split(",") if r.strip()]
    else:
        test_rounds = all_rounds

    results = {}

    for held_out in test_rounds:
        # Always train on ALL rounds except held-out (use full dataset)
        train_on = [r for r in all_rounds if r != held_out]
        train_on = [r for r in test_rounds if r != held_out]
        gbt_models = train_gbt_models(train_on)

        round_id = ROUNDS[held_out]
        initial_states, gts = load_round_data(held_out)
        all_observations = load_observations(round_id)

        # Build cross-seed empirical distance tables
        all_grids = [initial_states[s]["grid"] for s in range(5)]
        all_settl = [initial_states[s]["settlements"] for s in range(5)]
        round_empirical = build_round_empirical_tables(
            all_observations, all_grids, all_settlements=all_settl
        ) if all_observations else {}

        # Pre-compute cell-level obs features (shared across seeds)
        g0 = initial_states[0]["grid"]
        h0, w0 = len(g0), len(g0[0]) if g0 else 0
        cell_obs = compute_cell_obs_features(all_observations, h0, w0) if all_observations else None
        obs_stats = compute_obs_stats(all_observations) if all_observations else None

        scores = []
        for seed in range(5):
            grid = initial_states[seed]["grid"]
            settlements = initial_states[seed]["settlements"]
            gt = gts[seed]

            # Layer 1+3: Static + context priors
            tensor = build_static_prediction(grid)
            tensor = fill_unobserved_dynamic(tensor, grid, settlements, [], seed)

            # Layer 6: GBT blend (vectorized)
            has_obs = bool(all_observations)
            gbt_pred = gbt_predict_with_models(gbt_models, grid, settlements, obs_stats=obs_stats, cell_obs=cell_obs) if has_obs else None
            if gbt_pred is not None:
                grid_arr = np.array(grid)
                h, w, _ = tensor.shape
                # Build per-cell blend weight array
                bw_arr = np.zeros((h, w))
                for ttype, codes in [("plains", [11, 0]), ("forest", [4]), ("settl", [1, 2])]:
                    mask = np.isin(grid_arr, codes)
                    bw_arr[mask] = BLEND_TERRAIN.get(ttype, BLEND_WEIGHT)
                dynamic = ~np.isin(grid_arr, [10, 5])
                bw_3d = bw_arr[..., np.newaxis]
                tensor[dynamic] = (1 - bw_3d[dynamic]) * tensor[dynamic] + bw_3d[dynamic] * gbt_pred[dynamic]

            # Layer 6.7: Cross-seed empirical distance tables
            if all_observations and round_empirical:
                h, w, _ = tensor.shape
                EMP_BLEND = EMP_BLEND_EXPANSION if classify_round_type(obs_stats) == "expansion" else EMP_BLEND_NORMAL
                grid_arr = np.array(grid)
                settl_pos = [(s['x'] if isinstance(s, dict) else s.x,
                              s['y'] if isinstance(s, dict) else s.y) for s in settlements]
                for y in range(h):
                    for x in range(w):
                        code = grid[y][x]
                        if code in {10, 5}:
                            continue
                        dist = min((abs(y-sy)+abs(x-sx) for sx,sy in settl_pos), default=99)
                        dist_bucket = _dist_to_bucket(dist)
                        coastal = _is_coastal_4connected(grid, y, x)
                        key = (code, dist_bucket, coastal)
                        if key in round_empirical:
                            tensor[y, x] = (1 - EMP_BLEND) * tensor[y, x] + EMP_BLEND * round_empirical[key]

            # Layer 7: Round-type-adaptive observation ratio correction (vectorized)
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
                    grid_arr = np.array(grid)
                    dynamic = ~np.isin(grid_arr, [10, 5])
                    model_avg = tensor[dynamic].mean(axis=0)
                    ratio = obs_freq / np.maximum(model_avg, 1e-6)
                    # Use round-type-adaptive L7 parameters
                    round_type = classify_round_type(obs_stats)
                    rt_params = ROUND_TYPE_L7[round_type]
                    strengths = rt_params["strengths"].copy()
                    for c in range(NUM_CLASSES):
                        if obs_cls[c] < L7_MIN_OBS[c]:
                            strengths[c] = 0.0
                    adj = 1.0 + strengths * (ratio - 1.0)
                    adj_min = rt_params["adj_min"]
                    adj_max = rt_params["adj_max"]
                    adj = np.clip(adj, adj_min, adj_max)
                    # Apply to all dynamic cells at once
                    tensor[dynamic] *= adj[np.newaxis, :]
                    tensor[dynamic] = np.maximum(tensor[dynamic], PROB_FLOOR)
                    tensor[dynamic] /= tensor[dynamic].sum(axis=1, keepdims=True)

            # Layer 8: Per-cell empirical (effectively disabled with MIN_SAMPLES=50)
            if all_observations:
                h, w, _ = tensor.shape
                cell_counts = np.zeros((h, w), dtype=np.int32)
                cell_terrain = np.zeros((h, w, NUM_CLASSES))
                for obs in all_observations:
                    vp = obs.get("viewport", {})
                    vy, vx = vp.get("y", 0), vp.get("x", 0)
                    obs_grid = obs.get("grid", [])
                    for dy in range(len(obs_grid)):
                        for dx in range(len(obs_grid[0]) if obs_grid else 0):
                            y2, x2 = vy + dy, vx + dx
                            if 0 <= y2 < h and 0 <= x2 < w:
                                cell_counts[y2, x2] += 1
                                cls = TERRAIN_TO_CLASS.get(obs_grid[dy][dx], 0)
                                cell_terrain[y2, x2, cls] += 1

                MIN_SAMPLES = 50
                grid_arr = np.array(grid)
                dynamic = ~np.isin(grid_arr, [10, 5])
                enough = (cell_counts >= MIN_SAMPLES) & dynamic
                if enough.any():
                    EMP_WEIGHT_PER_SAMPLE = 0.03
                    MAX_EMP_WEIGHT = 0.30
                    n = cell_counts[enough, np.newaxis]
                    emp = cell_terrain[enough] / n
                    emp = np.maximum(emp, PROB_FLOOR)
                    emp /= emp.sum(axis=1, keepdims=True)
                    alpha = np.minimum(MAX_EMP_WEIGHT, cell_counts[enough] * EMP_WEIGHT_PER_SAMPLE)[:, np.newaxis]
                    tensor[enough] = (1 - alpha) * tensor[enough] + alpha * emp
                    tensor[enough] = np.maximum(tensor[enough], PROB_FLOOR)
                    tensor[enough] /= tensor[enough].sum(axis=1, keepdims=True)

            # Spatial smoothing DISABLED for isolation test
            # tensor = spatial_smooth(tensor, grid)
            tensor = normalize_prediction(tensor)
            score = compute_score(tensor, gt)
            scores.append(score)

        results[held_out] = np.mean(scores)

    return np.mean(list(results.values())), results


if __name__ == "__main__":
    start = time.time()

    avg, per_round = evaluate_loro()

    elapsed = time.time() - start

    # Per-round diagnostics
    for r, s in sorted(per_round.items()):
        print(f"round_{r}_score: {s:.4f}")

    # Weighted average (competition metric: 1.05^(round-1))
    weights = ROUND_WEIGHTS
    w_avg = sum(per_round[r] * weights[r] for r in per_round) / sum(weights[r] for r in per_round)
    print(f"weighted_avg: {w_avg:.4f}")

    # Autoresearch-compatible output (val_metric = weighted avg for competition alignment)
    print("---")
    print(f"val_metric: {w_avg:.6f}")
    print(f"val_metric_unweighted: {avg:.6f}")
    print(f"training_seconds: {elapsed:.1f}")
    print(f"total_seconds: {elapsed:.1f}")
    print(f"peak_vram_mb: 0.0")
