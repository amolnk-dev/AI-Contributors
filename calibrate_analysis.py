"""Calibration analysis: replay dynamics + grid search for optimal simulator params."""

import json
import sys
import time
import numpy as np
from collections import defaultdict

from simulator import (
    OCEAN, PLAINS, SETTLEMENT, PORT, RUIN, FOREST, MOUNTAIN,
    simulate_monte_carlo,
)
from evaluate import compute_score


def pprint(*args, **kwargs):
    """Print with flush for immediate output."""
    print(*args, **kwargs, flush=True)


TERRAIN_NAMES = {
    OCEAN: "ocean", PLAINS: "plains", SETTLEMENT: "settlement",
    PORT: "port", RUIN: "ruin", FOREST: "forest", MOUNTAIN: "mountain",
}


def analyze_replay(replay_path):
    """Compute per-step dynamics from a replay file."""
    with open(replay_path) as f:
        data = json.load(f)

    frames = data["frames"]
    results = {
        "births": [],
        "deaths": [],
        "ruin_reclamations": [],
        "new_ports": [],
        "expand_terrain": defaultdict(int),
        "expand_distances": [],
        "alive_counts": [],
        "ruin_counts": [],
        "forest_counts": [],
        "port_counts": [],
        "settlement_counts": [],
    }

    for i in range(len(frames)):
        grid = np.array(frames[i]["grid"])
        settlements = frames[i]["settlements"]

        alive = sum(1 for s in settlements if s["alive"])
        results["alive_counts"].append(alive)
        results["ruin_counts"].append(int(np.sum(grid == RUIN)))
        results["forest_counts"].append(int(np.sum(grid == FOREST)))
        results["port_counts"].append(int(np.sum(grid == PORT)))
        results["settlement_counts"].append(int(np.sum(grid == SETTLEMENT)))

        if i == 0:
            continue

        prev_grid = np.array(frames[i - 1]["grid"])

        was_active = np.isin(prev_grid, [SETTLEMENT, PORT])
        is_active = np.isin(grid, [SETTLEMENT, PORT])
        new_active = is_active & ~was_active
        births = int(np.sum(new_active))
        results["births"].append(births)

        became_ruin = (grid == RUIN) & was_active
        deaths = int(np.sum(became_ruin))
        results["deaths"].append(deaths)

        was_ruin = (prev_grid == RUIN)
        ruin_gone = was_ruin & (grid != RUIN)
        ruin_reclamations = int(np.sum(ruin_gone))
        results["ruin_reclamations"].append(ruin_reclamations)

        new_ports = int(np.sum((grid == PORT) & (prev_grid != PORT)))
        results["new_ports"].append(new_ports)

        new_positions = np.argwhere(new_active)
        if len(new_positions) > 0:
            for idx in range(len(new_positions)):
                ny, nx = new_positions[idx]
                prev_terrain = prev_grid[ny, nx]
                results["expand_terrain"][prev_terrain] += 1

            prev_settle_positions = np.argwhere(was_active)
            if len(prev_settle_positions) > 0:
                diffs = np.abs(
                    new_positions[:, np.newaxis, :] - prev_settle_positions[np.newaxis, :, :]
                )
                manhattan = diffs.sum(axis=2)
                min_dists = manhattan.min(axis=1)
                results["expand_distances"].extend(min_dists.tolist())

    return results


def main():
    # ========================================================================
    # PART 1: Per-step dynamics from replays
    # ========================================================================
    pprint("=" * 90)
    pprint("PART 1: PER-REPLAY DYNAMICS")
    pprint("=" * 90)

    all_results = {}
    for r in range(1, 6):
        for s in range(5):
            path = f"data/replay_r{r}_seed{s}.json"
            key = f"r{r}_s{s}"
            all_results[key] = analyze_replay(path)
        pprint(f"  Round {r} replays analyzed.")

    pprint(f"\n{'Replay':<12} {'TotBirths':>10} {'TotDeaths':>10} {'TotReclaim':>10} "
          f"{'TotPorts':>10} {'FinalAlive':>10} {'FinalRuin':>10}")
    pprint("-" * 74)
    for key in sorted(all_results.keys()):
        res = all_results[key]
        pprint(f"{key:<12} {sum(res['births']):>10} {sum(res['deaths']):>10} "
              f"{sum(res['ruin_reclamations']):>10} {sum(res['new_ports']):>10} "
              f"{res['alive_counts'][-1]:>10} {res['ruin_counts'][-1]:>10}")

    # ========================================================================
    # PART 2: Per-round average dynamics
    # ========================================================================
    pprint("\n" + "=" * 90)
    pprint("PART 2: PER-ROUND AVERAGE DYNAMICS (across 5 seeds)")
    pprint("=" * 90)

    for r in range(1, 6):
        seed_keys = [f"r{r}_s{s}" for s in range(5)]

        expansion_rates = []
        death_rates = []
        for key in seed_keys:
            res = all_results[key]
            for step_idx in range(len(res["births"])):
                alive = res["alive_counts"][step_idx]
                if alive > 0:
                    expansion_rates.append(res["births"][step_idx] / alive)
                    death_rates.append(res["deaths"][step_idx] / alive)

        avg_expansion = np.mean(expansion_rates) if expansion_rates else 0
        avg_death = np.mean(death_rates) if death_rates else 0

        final_alive = np.mean([all_results[k]["alive_counts"][-1] for k in seed_keys])
        final_ruin = np.mean([all_results[k]["ruin_counts"][-1] for k in seed_keys])
        final_forest = np.mean([all_results[k]["forest_counts"][-1] for k in seed_keys])
        final_port = np.mean([all_results[k]["port_counts"][-1] for k in seed_keys])
        final_settl = np.mean([all_results[k]["settlement_counts"][-1] for k in seed_keys])

        total_births = sum(sum(all_results[k]["births"]) for k in seed_keys)
        total_ports = sum(sum(all_results[k]["new_ports"]) for k in seed_keys)
        port_rate = total_ports / total_births if total_births > 0 else 0

        terrain_counts = defaultdict(int)
        for key in seed_keys:
            for t, c in all_results[key]["expand_terrain"].items():
                terrain_counts[t] += c
        total_expand = sum(terrain_counts.values())

        all_dists = []
        for key in seed_keys:
            all_dists.extend(all_results[key]["expand_distances"])
        dist_counts = defaultdict(int)
        for d in all_dists:
            dist_counts[d] += 1

        pprint(f"\n--- Round {r} ---")
        pprint(f"  Avg expansion rate:  {avg_expansion:.4f} (births/step / alive)")
        pprint(f"  Avg death rate:      {avg_death:.4f} (deaths/step / alive)")
        pprint(f"  Final alive (avg):   {final_alive:.1f}")
        pprint(f"  Final settlements:   {final_settl:.1f}")
        pprint(f"  Final ports:         {final_port:.1f}")
        pprint(f"  Final ruins:         {final_ruin:.1f}")
        pprint(f"  Final forests:       {final_forest:.1f}")
        pprint(f"  Port formation rate: {port_rate:.4f}")

        if total_expand > 0:
            parts = []
            for t in sorted(terrain_counts.keys()):
                name = TERRAIN_NAMES.get(t, f"code{t}")
                pct = terrain_counts[t] / total_expand * 100
                parts.append(f"{name}={pct:.1f}%")
            pprint(f"  Expansion terrain: {' '.join(parts)}")

        if all_dists:
            parts = []
            for d in sorted(dist_counts.keys()):
                pct = dist_counts[d] / len(all_dists) * 100
                parts.append(f"d{d}={pct:.1f}%")
            pprint(f"  Expansion distances: {' '.join(parts)}")

    # ========================================================================
    # PART 3: Observable terrain frequencies from final replay frames
    # ========================================================================
    pprint("\n" + "=" * 90)
    pprint("PART 3: OBSERVABLE TERRAIN FREQUENCIES (from final replay frames)")
    pprint("=" * 90)

    for r in range(1, 6):
        obs_settl_rates = []
        obs_ruin_rates = []
        obs_forest_rates = []
        for s in range(5):
            path = f"data/replay_r{r}_seed{s}.json"
            with open(path) as f:
                replay = json.load(f)
            final_grid = np.array(replay["frames"][-1]["grid"])

            dynamic = ~np.isin(final_grid, [OCEAN, MOUNTAIN])
            n_dynamic = np.sum(dynamic)
            if n_dynamic > 0:
                obs_settl_rates.append(
                    np.sum(np.isin(final_grid, [SETTLEMENT, PORT]) & dynamic) / n_dynamic
                )
                obs_ruin_rates.append(np.sum((final_grid == RUIN) & dynamic) / n_dynamic)
                obs_forest_rates.append(np.sum((final_grid == FOREST) & dynamic) / n_dynamic)

        pprint(f"  Round {r}: obs_settl={np.mean(obs_settl_rates):.4f}  "
              f"obs_ruin={np.mean(obs_ruin_rates):.4f}  "
              f"obs_forest={np.mean(obs_forest_rates):.4f}")

    # ========================================================================
    # PART 4: Summary table
    # ========================================================================
    pprint("\n" + "=" * 90)
    pprint("PART 4: SUMMARY TABLE")
    pprint("=" * 90)

    pprint(f"\n{'Round':<8} {'ExpRate':>8} {'DeathRate':>10} {'ObsSettl':>10} {'ObsRuin':>10} "
          f"{'ObsForest':>10} {'FinalAlive':>11} {'FinalRuin':>10} {'FinalPort':>10}")
    pprint("-" * 98)

    for r in range(1, 6):
        seed_keys = [f"r{r}_s{s}" for s in range(5)]

        expansion_rates = []
        death_rates = []
        for key in seed_keys:
            res = all_results[key]
            for step_idx in range(len(res["births"])):
                alive = res["alive_counts"][step_idx]
                if alive > 0:
                    expansion_rates.append(res["births"][step_idx] / alive)
                    death_rates.append(res["deaths"][step_idx] / alive)

        avg_expansion = np.mean(expansion_rates)
        avg_death = np.mean(death_rates)

        obs_s, obs_r, obs_f = [], [], []
        for s in range(5):
            path = f"data/replay_r{r}_seed{s}.json"
            with open(path) as f_:
                replay = json.load(f_)
            final_grid = np.array(replay["frames"][-1]["grid"])
            dynamic = ~np.isin(final_grid, [OCEAN, MOUNTAIN])
            n_dyn = np.sum(dynamic)
            if n_dyn > 0:
                obs_s.append(np.sum(np.isin(final_grid, [SETTLEMENT, PORT]) & dynamic) / n_dyn)
                obs_r.append(np.sum((final_grid == RUIN) & dynamic) / n_dyn)
                obs_f.append(np.sum((final_grid == FOREST) & dynamic) / n_dyn)

        final_alive = np.mean([all_results[k]["alive_counts"][-1] for k in seed_keys])
        final_ruin = np.mean([all_results[k]["ruin_counts"][-1] for k in seed_keys])
        final_port = np.mean([all_results[k]["port_counts"][-1] for k in seed_keys])

        pprint(f"  R{r:<5} {avg_expansion:>8.4f} {avg_death:>10.4f} {np.mean(obs_s):>10.4f} "
              f"{np.mean(obs_r):>10.4f} {np.mean(obs_f):>10.4f} {final_alive:>11.1f} "
              f"{final_ruin:>10.1f} {final_port:>10.1f}")

    # ========================================================================
    # PART 5: Grid search for optimal simulator params (seed 0 only, n_sims=20)
    # ========================================================================
    pprint("\n" + "=" * 90)
    pprint("PART 5: GRID SEARCH FOR OPTIMAL PARAMS (seed 0, n_sims=20)")
    pprint("=" * 90)

    expansion_range = np.arange(0.04, 0.18, 0.01)
    winter_range = np.arange(0.03, 0.20, 0.01)

    pprint(f"\nGrid: expansion_rate in [{expansion_range[0]:.3f}, {expansion_range[-1]:.3f}] "
          f"({len(expansion_range)} values)")
    pprint(f"       winter_severity in [{winter_range[0]:.3f}, {winter_range[-1]:.3f}] "
          f"({len(winter_range)} values)")
    pprint(f"Total evaluations per round: {len(expansion_range) * len(winter_range)}")

    best_params = {}

    for r in range(1, 6):
        t0 = time.time()

        with open(f"data/round{r}_initial.json") as f:
            round_data = json.load(f)
        initial_grid = round_data["initial_states"][0]["grid"]
        initial_settlements = round_data["initial_states"][0]["settlements"]

        gt = np.load(f"data/gt_r{r}_seed0.npy")

        best_score = -1
        best_exp = None
        best_win = None
        scores_grid = np.zeros((len(expansion_range), len(winter_range)))

        total_evals = len(expansion_range) * len(winter_range)
        done = 0

        for ei, exp_rate in enumerate(expansion_range):
            for wi, win_sev in enumerate(winter_range):
                probs = simulate_monte_carlo(
                    initial_grid, initial_settlements,
                    n_sims=20,
                    expansion_rate=exp_rate,
                    winter_severity=win_sev,
                    seed=42,
                    n_workers=1,
                )
                score = compute_score(probs, gt)
                scores_grid[ei, wi] = score

                if score > best_score:
                    best_score = score
                    best_exp = exp_rate
                    best_win = win_sev

                done += 1

            elapsed_so_far = time.time() - t0
            pprint(f"    R{r} progress: {done}/{total_evals} "
                  f"({done/total_evals*100:.0f}%) "
                  f"best so far: exp={best_exp:.3f} win={best_win:.3f} score={best_score:.2f} "
                  f"[{elapsed_so_far:.0f}s]")

        elapsed = time.time() - t0
        best_params[r] = (best_exp, best_win, best_score)

        pprint(f"\n  Round {r}: best expansion_rate={best_exp:.3f}, winter_severity={best_win:.3f}, "
              f"score={best_score:.2f}  ({elapsed:.0f}s)")

        flat_idx = np.argsort(scores_grid.ravel())[::-1][:5]
        pprint(f"    Top 5 combos:")
        for rank, idx in enumerate(flat_idx):
            ei, wi = divmod(idx, len(winter_range))
            pprint(f"      #{rank+1}: exp={expansion_range[ei]:.3f}, win={winter_range[wi]:.3f}, "
                  f"score={scores_grid[ei, wi]:.2f}")

    # Final summary
    pprint("\n" + "=" * 90)
    pprint("GRID SEARCH SUMMARY")
    pprint("=" * 90)
    pprint(f"\n{'Round':<8} {'BestExpRate':>12} {'BestWinSev':>12} {'Score':>8}")
    pprint("-" * 42)
    for r in range(1, 6):
        exp, win, sc = best_params[r]
        pprint(f"  R{r:<5} {exp:>12.3f} {win:>12.3f} {sc:>8.2f}")

    avg_score = np.mean([best_params[r][2] for r in range(1, 6)])
    pprint(f"\n  Average score across rounds: {avg_score:.2f}")
    pprint(f"  Current default: expansion_rate=0.087, winter_severity=0.090")


if __name__ == "__main__":
    main()
