"""Norse civilization simulator for Astar Island — calibrated from 25 replay trajectories.

Key dynamics learned from replays (R1-R5, 5 seeds each):
  - Expansion: ~10% of alive settlements expand per step (higher for high-pop)
  - Deaths: ~7% of alive settlements die per step (stochastic, bursty)
  - Ruins reclaim in ~1 step: ~50% re-settled, ~35% plains, ~15% forest
  - Expansion targets: 45% plains, 38% ruins, 17% forest
  - Expansion distance: 41% d=1, 33% d=2, 21% d=3, 5% d=4
  - Port formation: coastal settlements (adj_ocean>=2) can become ports (~2%)
  - New settlements start with pop~0.45, food~0.22, defense~0.175

Usage:
    from simulator import simulate_monte_carlo, fit_hidden_params
    params = fit_hidden_params(initial_grid, settlements, observations)
    probs = simulate_monte_carlo(initial_grid, settlements, n_sims=500, **params)
"""

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Terrain codes
OCEAN = 10
PLAINS = 11
EMPTY = 0
SETTLEMENT = 1
PORT = 2
RUIN = 3
FOREST = 4
MOUNTAIN = 5

# Expandable terrain types
EXPANDABLE = frozenset({PLAINS, RUIN, FOREST})

# Expansion distance weights (from replay data)
DIST_WEIGHTS = np.array([0.0, 0.41, 0.33, 0.21, 0.05])

# Terrain preference weights for expansion targets
# Replay data: 37.5% plains, 47.8% ruins, 14.8% forest
# Ruins get high weight because they're near existing settlements
TERRAIN_EXPAND_WEIGHT = {PLAINS: 1.0, RUIN: 3.2, FOREST: 0.4}

# Reclamation: ruin -> plains (68%) or forest (32%)
RECLAIM_PLAINS_PROB = 0.68

# Pre-build offsets for Manhattan distance 1-4
_OFFSETS_BY_DIST = {}
for _d in range(1, 5):
    offsets = []
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            if abs(dy) + abs(dx) == _d:
                offsets.append((dy, dx))
    _OFFSETS_BY_DIST[_d] = offsets


def _count_neighbors(grid, code):
    """Count cells of given code in 8-neighborhood of each cell."""
    h, w = grid.shape
    mask = (grid == code).astype(np.int32)
    count = np.zeros((h, w), dtype=np.int32)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            sy = max(0, dy)
            ey = min(h, h + dy)
            sx = max(0, dx)
            ex = min(w, w + dx)
            fy = max(0, -dy)
            fx = max(0, -dx)
            count[sy:ey, sx:ex] += mask[fy:fy + (ey - sy), fx:fx + (ex - sx)]
    return count


def simulate_one(
    initial_grid,
    initial_settlements,
    expansion_rate=0.087,
    winter_severity=0.090,
    raid_intensity=0.03,
    rng=None,
):
    """Run one stochastic simulation for 50 steps.

    Phase ordering (calibrated from replay frame-by-frame analysis):
      1. Ruin reclamation (from previous step's deaths)
      2. Growth & resource gathering (vectorized)
      3. Port development (vectorized)
      4. Expansion (new settlements)
      5. Conflict (raiding)
      6. Winter / death (vectorized)

    Returns the final H x W terrain grid.
    """
    if rng is None:
        rng = np.random.default_rng()

    h = len(initial_grid)
    w = len(initial_grid[0]) if h > 0 else 0
    grid = np.array(initial_grid, dtype=np.int32)

    # Static features (don't change during sim)
    ocean_nbrs = _count_neighbors(grid, OCEAN)

    # Pre-allocate settlement arrays
    n_init = len(initial_settlements)
    max_settle = n_init + 2000
    s_x = np.zeros(max_settle, dtype=np.int32)
    s_y = np.zeros(max_settle, dtype=np.int32)
    s_pop = np.zeros(max_settle, dtype=np.float64)
    s_food = np.zeros(max_settle, dtype=np.float64)
    s_def = np.zeros(max_settle, dtype=np.float64)
    s_port = np.zeros(max_settle, dtype=bool)
    s_alive = np.zeros(max_settle, dtype=bool)
    s_owner = np.zeros(max_settle, dtype=np.int32)

    for i, s in enumerate(initial_settlements):
        if isinstance(s, dict):
            s_x[i], s_y[i] = s.get('x', 0), s.get('y', 0)
            s_port[i] = s.get('has_port', False)
            s_owner[i] = s.get('owner_id', i)
        else:
            s_x[i], s_y[i] = s.x, s.y
            s_port[i] = getattr(s, 'has_port', False)
            s_owner[i] = getattr(s, 'owner_id', i)
        s_pop[i] = rng.uniform(0.6, 1.3)
        s_food[i] = rng.uniform(0.4, 0.8)
        s_def[i] = rng.uniform(0.2, 0.6)
        s_alive[i] = True

    n_settle = n_init
    # Grid-based occupancy for O(1) lookups
    occ_grid = np.full((h, w), -1, dtype=np.int32)
    for i in range(n_settle):
        occ_grid[s_y[i], s_x[i]] = i

    def _add(x, y, p, f, d, hp, oid):
        nonlocal n_settle
        if n_settle >= max_settle:
            return
        s_x[n_settle] = x
        s_y[n_settle] = y
        s_pop[n_settle] = p
        s_food[n_settle] = f
        s_def[n_settle] = d
        s_port[n_settle] = hp
        s_alive[n_settle] = True
        s_owner[n_settle] = oid
        occ_grid[y, x] = n_settle
        n_settle += 1

    for step in range(50):
        alive_idx = np.where(s_alive[:n_settle])[0]
        n_alive = len(alive_idx)
        if n_alive == 0:
            break

        # === PHASE 1: RUIN RECLAMATION ===
        ruin_ys, ruin_xs = np.where(grid == RUIN)
        if len(ruin_ys) > 0:
            perm = rng.permutation(len(ruin_ys))
            for pi in perm:
                ry, rx = int(ruin_ys[pi]), int(ruin_xs[pi])
                if grid[ry, rx] != RUIN:
                    continue
                # Find nearby alive settlements via grid lookup
                nearby = []
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        d = abs(dy) + abs(dx)
                        if d == 0 or d > 3:
                            continue
                        ny, nx = ry + dy, rx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            si = occ_grid[ny, nx]
                            if si >= 0 and s_alive[si]:
                                nearby.append(si)

                if nearby and rng.random() < 0.457:
                    # 45.7% of ruins get re-settled (from replay data)
                    parent = nearby[rng.integers(len(nearby))]
                    is_port = ocean_nbrs[ry, rx] >= 2 and rng.random() < 0.01
                    grid[ry, rx] = PORT if is_port else SETTLEMENT
                    _add(rx, ry, rng.uniform(0.35, 0.55), rng.uniform(0.10, 0.25),
                         rng.uniform(0.12, 0.18), is_port, s_owner[parent])
                elif rng.random() < 0.543:
                    # Remaining ruins: 36.8% → plains, 17.5% → forest (normalized)
                    grid[ry, rx] = PLAINS if rng.random() < 0.678 else FOREST

        # Refresh alive indices
        alive_idx = np.where(s_alive[:n_settle])[0]
        n_alive = len(alive_idx)

        # === PHASE 2: GROWTH (vectorized) ===
        forest_nbrs = _count_neighbors(grid, FOREST)
        ai_y = s_y[alive_idx]
        ai_x = s_x[alive_idx]
        adj_f = forest_nbrs[ai_y, ai_x].astype(np.float64)
        s_food[alive_idx] = np.minimum(1.0, s_food[alive_idx] + adj_f * 0.10)

        grow_mask = s_food[alive_idx] > 0.3
        grow_idx = alive_idx[grow_mask]
        n_grow = len(grow_idx)
        if n_grow > 0:
            s_pop[grow_idx] = np.minimum(
                3.0,
                s_pop[grow_idx] + 0.07 * s_food[grow_idx] * rng.uniform(0.3, 1.7, size=n_grow)
            )

        s_def[alive_idx] = np.minimum(
            1.0,
            s_def[alive_idx] + 0.015 * rng.uniform(0.5, 1.5, size=n_alive)
        )

        # === PHASE 3: PORT DEVELOPMENT (vectorized) ===
        non_port_mask = ~s_port[alive_idx]
        non_port_idx = alive_idx[non_port_mask]
        if len(non_port_idx) > 0:
            oc = ocean_nbrs[s_y[non_port_idx], s_x[non_port_idx]]
            p_port = np.where(oc >= 2, 0.10, np.where(oc >= 1, 0.03, 0.0)) * s_pop[non_port_idx]
            rolls = rng.random(len(non_port_idx))
            became_port = non_port_idx[rolls < p_port]
            for i in became_port:
                s_port[i] = True
                grid[s_y[i], s_x[i]] = PORT

        # === PHASE 4: EXPANSION ===
        expand_order = alive_idx.copy()
        rng.shuffle(expand_order)

        for i in expand_order:
            if s_pop[i] < 0.4:
                continue
            p_expand = expansion_rate * (0.5 + 0.5 * s_pop[i])
            if rng.random() > p_expand:
                continue

            candidates = []
            weights = []
            sy_i, sx_i = s_y[i], s_x[i]
            for d in range(1, 5):
                dw = DIST_WEIGHTS[d]
                for dy, dx in _OFFSETS_BY_DIST[d]:
                    ny, nx = sy_i + dy, sx_i + dx
                    if 0 <= ny < h and 0 <= nx < w and occ_grid[ny, nx] < 0:
                        t = grid[ny, nx]
                        if t in EXPANDABLE:
                            candidates.append((ny, nx))
                            weights.append(TERRAIN_EXPAND_WEIGHT.get(t, 0) * dw)

            if not candidates:
                continue

            wt = np.array(weights)
            wt /= wt.sum()
            idx = rng.choice(len(candidates), p=wt)
            ty, tx = candidates[idx]

            is_port_new = ocean_nbrs[ty, tx] >= 2 and rng.random() < 0.01
            grid[ty, tx] = PORT if is_port_new else SETTLEMENT
            _add(tx, ty, rng.uniform(0.35, 0.55), rng.uniform(0.15, 0.30),
                 rng.uniform(0.14, 0.20), is_port_new, s_owner[i])
            s_pop[i] *= 0.80
            s_food[i] *= 0.75

        # === PHASE 5: CONFLICT ===
        alive_idx = np.where(s_alive[:n_settle])[0]
        for i in alive_idx:
            if not s_alive[i]:
                continue
            desperation = max(0, 1.0 - s_food[i])
            p_raid = raid_intensity * (1.0 + desperation * 1.5)
            if rng.random() > p_raid:
                continue

            max_range = 5 if s_port[i] else 3
            enemies = []
            for j in alive_idx:
                if j != i and s_owner[j] != s_owner[i] and s_alive[j]:
                    dist = abs(s_x[i] - s_x[j]) + abs(s_y[i] - s_y[j])
                    if 0 < dist <= max_range:
                        enemies.append(j)

            if not enemies:
                continue

            j = enemies[rng.integers(len(enemies))]
            attack = s_pop[i] * (0.5 + s_def[i]) * rng.uniform(0.4, 1.6)
            defend_str = s_pop[j] * (0.5 + s_def[j]) * rng.uniform(0.4, 1.6)

            if attack > defend_str:
                loot = min(s_food[j] * 0.3, 0.2)
                s_food[i] = min(1.0, s_food[i] + loot)
                s_food[j] -= loot
                s_pop[j] *= 0.65
                s_def[j] *= 0.75

        # === PHASE 6: WINTER / DEATH (vectorized) ===
        winter_mult = rng.uniform(0.1, 3.0)
        alive_idx = np.where(s_alive[:n_settle])[0]
        n_alive = len(alive_idx)
        if n_alive == 0:
            break

        # Vectorized food/pop drain
        food_drain = winter_severity * winter_mult * rng.uniform(0.3, 1.7, size=n_alive)
        s_food[alive_idx] -= food_drain
        s_food[alive_idx] = np.maximum(s_food[alive_idx], -0.3)

        pop_drain = winter_severity * winter_mult * 0.15 * rng.uniform(0.0, 1.5, size=n_alive)
        s_pop[alive_idx] -= pop_drain

        # Vectorized death probability
        p_die = np.full(n_alive, winter_severity * winter_mult * 0.20)
        food_v = s_food[alive_idx]
        pop_v = s_pop[alive_idx]
        def_v = s_def[alive_idx]

        p_die += np.where(food_v < 0.0, 0.15,
                 np.where(food_v < 0.3, 0.08,
                 np.where(food_v < 0.5, 0.03, 0.0)))
        p_die += np.where(pop_v < 0.3, 0.12,
                 np.where(pop_v < 0.5, 0.06,
                 np.where(pop_v < 0.8, 0.02, 0.0)))
        p_die += np.where(def_v < 0.15, 0.02, 0.0)

        die_rolls = rng.random(n_alive)
        die_mask = die_rolls < p_die
        die_idx = alive_idx[die_mask]
        for i in die_idx:
            s_alive[i] = False
            grid[s_y[i], s_x[i]] = RUIN
            occ_grid[s_y[i], s_x[i]] = -1

    return grid


# ── Multiprocessing support ────────────────────────────────────

_worker_data = None


def _init_worker(grid, settlements):
    """Initialize shared data in worker process."""
    global _worker_data
    _worker_data = (grid, settlements)


def _sim_worker(args):
    """Worker function for parallel simulation."""
    er, ws, ri, sim_seed = args
    grid, settlements = _worker_data
    rng = np.random.default_rng(sim_seed)
    return simulate_one(grid, settlements, er, ws, ri, rng)


def simulate_monte_carlo(
    initial_grid,
    initial_settlements,
    n_sims=100,
    expansion_rate=0.087,
    winter_severity=0.090,
    raid_intensity=0.03,
    seed=42,
    n_workers=None,
):
    """Run Monte Carlo simulations and return probability distributions.

    Uses multiprocessing for parallel execution on multiple cores.
    Returns H x W x 6 tensor of terrain class probabilities.
    """
    h = len(initial_grid)
    w = len(initial_grid[0]) if h > 0 else 0

    # Class mapping lookup table for vectorized counting
    code_to_cls = np.zeros(12, dtype=np.int32)
    code_to_cls[0] = 0    # Empty
    code_to_cls[10] = 0   # Ocean
    code_to_cls[11] = 0   # Plains
    code_to_cls[1] = 1    # Settlement
    code_to_cls[2] = 2    # Port
    code_to_cls[3] = 3    # Ruin
    code_to_cls[4] = 4    # Forest
    code_to_cls[5] = 5    # Mountain

    counts = np.zeros((h, w, 6), dtype=np.float64)

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 6)

    args_list = [
        (expansion_rate, winter_severity, raid_intensity, seed * 10000 + i)
        for i in range(n_sims)
    ]

    if n_workers > 1 and n_sims >= 8:
        # Parallel execution
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(initial_grid, initial_settlements),
        ) as executor:
            chunksize = max(1, n_sims // (n_workers * 4))
            results = list(executor.map(_sim_worker, args_list, chunksize=chunksize))
    else:
        # Sequential for small runs (avoids multiprocessing overhead)
        results = []
        for args in args_list:
            er, ws, ri, sim_seed = args
            rng = np.random.default_rng(sim_seed)
            results.append(simulate_one(
                initial_grid, initial_settlements, er, ws, ri, rng
            ))

    # Vectorized counting
    for final_grid in results:
        cls_grid = code_to_cls[final_grid]
        for cls in range(6):
            counts[:, :, cls] += (cls_grid == cls)

    probs = counts / n_sims
    probs = np.maximum(probs, 0.0001)
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs


# ── Replay-calibrated parameter estimation ─────────────────────

# Calibration table: maps (obs_settl_rate, obs_ruin_rate) to optimal params.
# Built by grid-searching simulator params against 25 replay trajectories.
# Each entry: (obs_settl_lower, obs_settl_upper, obs_ruin_threshold,
#              expansion_rate, winter_severity)
# Will be populated after calibration analysis completes.
_CALIBRATION_TABLE = None


def _load_calibration_table():
    """Load pre-computed calibration from file if available."""
    global _CALIBRATION_TABLE
    if _CALIBRATION_TABLE is not None:
        return _CALIBRATION_TABLE

    cal_path = os.path.join(os.path.dirname(__file__), "data", "sim_calibration.json")
    if os.path.exists(cal_path):
        import json
        with open(cal_path) as f:
            _CALIBRATION_TABLE = json.load(f)
        return _CALIBRATION_TABLE
    return None


def fit_hidden_params(
    initial_grid,
    initial_settlements,
    observations,
    n_sims_per_eval=20,
):
    """Estimate hidden parameters from observation terrain frequencies.

    Uses replay-calibrated interpolation: maps observed settlement/ruin density
    to the simulator params that best matched replays with similar distributions.

    Falls back to a refined lookup table when calibration data isn't available.
    """
    from dtos import TERRAIN_TO_CLASS

    obs_cls = np.zeros(6)
    obs_total = 0
    for obs in observations:
        for row in obs.get("grid", []):
            for code in row:
                if code not in {10, 5}:
                    obs_cls[TERRAIN_TO_CLASS.get(code, 0)] += 1
                    obs_total += 1

    if obs_total < 50:
        return {"expansion_rate": 0.087, "winter_severity": 0.090, "raid_intensity": 0.03}

    obs_freq = obs_cls / obs_total
    obs_settl_rate = obs_freq[1] + obs_freq[2]  # settlements + ports
    obs_ruin_rate = obs_freq[3]
    obs_forest_rate = obs_freq[4]

    # Try calibration table first
    cal = _load_calibration_table()
    if cal and "rounds" in cal:
        # Interpolate from per-round calibration data
        # Find the two rounds with obs_settl_rate closest to ours
        rounds = cal["rounds"]
        best_dist = float("inf")
        best_er = 0.087
        best_ws = 0.090

        # Weighted average of all round params, weighted by similarity
        total_w = 0.0
        er_sum = 0.0
        ws_sum = 0.0
        for rdata in rounds:
            r_sr = rdata["obs_settl_rate"]
            r_rr = rdata["obs_ruin_rate"]
            # Distance metric combining settlement and ruin rates
            dist = abs(obs_settl_rate - r_sr) + 0.5 * abs(obs_ruin_rate - r_rr)
            if dist < 1e-8:
                dist = 1e-8
            w = 1.0 / (dist ** 2)
            er_sum += w * rdata["expansion_rate"]
            ws_sum += w * rdata["winter_severity"]
            total_w += w

        if total_w > 0:
            expansion_rate = er_sum / total_w
            winter_severity = ws_sum / total_w
            return {
                "expansion_rate": round(float(expansion_rate), 4),
                "winter_severity": round(float(winter_severity), 4),
                "raid_intensity": 0.03,
            }

    # Fallback: refined lookup table with linear interpolation
    # Anchored by replay analysis (will be updated with exact values)
    table = [
        # (obs_settl_rate, expansion_rate, winter_severity)
        (0.00, 0.040, 0.200),
        (0.03, 0.055, 0.150),
        (0.06, 0.068, 0.115),
        (0.09, 0.078, 0.098),
        (0.12, 0.087, 0.090),
        (0.15, 0.095, 0.080),
        (0.18, 0.105, 0.070),
        (0.22, 0.115, 0.060),
        (0.28, 0.130, 0.050),
    ]

    # Linear interpolation
    if obs_settl_rate <= table[0][0]:
        expansion_rate = table[0][1]
        winter_severity = table[0][2]
    elif obs_settl_rate >= table[-1][0]:
        expansion_rate = table[-1][1]
        winter_severity = table[-1][2]
    else:
        for j in range(len(table) - 1):
            if table[j][0] <= obs_settl_rate <= table[j + 1][0]:
                frac = (obs_settl_rate - table[j][0]) / (table[j + 1][0] - table[j][0])
                expansion_rate = table[j][1] + frac * (table[j + 1][1] - table[j][1])
                winter_severity = table[j][2] + frac * (table[j + 1][2] - table[j][2])
                break

    # Ruin rate adjustment
    if obs_ruin_rate > 0.015:
        winter_severity += 0.015 * min(obs_ruin_rate / 0.03, 2.0)

    return {
        "expansion_rate": round(expansion_rate, 4),
        "winter_severity": round(winter_severity, 4),
        "raid_intensity": 0.03,
    }


if __name__ == "__main__":
    import json
    import time
    from evaluate import compute_score

    with open("data/round4_initial.json") as f:
        r4 = json.load(f)

    grid = r4["initial_states"][0]["grid"]
    settlements = r4["initial_states"][0]["settlements"]
    gt = np.load("data/gt_r4_seed0.npy")

    print("Testing simulator on R4 seed 0:")
    print("=" * 70)

    for n_sims in [100, 200, 500]:
        t0 = time.time()
        probs = simulate_monte_carlo(
            grid, settlements, n_sims=n_sims,
            expansion_rate=0.087, winter_severity=0.090, seed=42,
        )
        elapsed = time.time() - t0
        score = compute_score(probs, gt)
        print(f"n={n_sims:4d}: score={score:.2f} "
              f"settl={probs[:,:,1].sum():.0f}/{gt[:,:,1].sum():.0f} "
              f"port={probs[:,:,2].sum():.1f}/{gt[:,:,2].sum():.1f} "
              f"ruin={probs[:,:,3].sum():.0f}/{gt[:,:,3].sum():.0f} "
              f"forest={probs[:,:,4].sum():.0f}/{gt[:,:,4].sum():.0f} "
              f"({elapsed:.1f}s)")
