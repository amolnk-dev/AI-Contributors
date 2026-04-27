"""Grid search calibration: find optimal (expansion_rate, winter_severity) per round.

Two-phase search:
  1. Coarse grid (step=0.03) with n_sims=20 to identify promising region
  2. Fine grid (step=0.01) around the best coarse result with n_sims=30

Writes results to data/sim_calibration.json for use by fit_hidden_params().
"""

import json
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from simulator import simulate_monte_carlo
from evaluate import compute_score
from dtos import TERRAIN_TO_CLASS
from utils import load_observations

# Round metadata
ROUNDS = {
    1: {"round_id": "71451d74-be9f-471f-aacd-a41f3b68a9cd"},
    2: {"round_id": "76909e29-f664-4b2f-b16b-61b7507277e9"},
    4: {"round_id": "8e839974-b13b-407b-a5e7-fc749d877195"},
    5: {"round_id": "fd3c92ff-3178-4dc9-8d9b-acf389b3982b"},
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def compute_obs_rates(round_id):
    """Compute observed settlement and ruin rates from observation data."""
    observations = load_observations(round_id)
    if not observations:
        return None, None

    obs_cls = np.zeros(6)
    obs_total = 0
    for obs in observations:
        for row in obs.get("grid", []):
            for code in row:
                if code not in {10, 5}:
                    cls = TERRAIN_TO_CLASS.get(code, 0)
                    obs_cls[cls] += 1
                    obs_total += 1

    if obs_total < 50:
        return None, None

    obs_freq = obs_cls / obs_total
    obs_settl_rate = obs_freq[1] + obs_freq[2]
    obs_ruin_rate = obs_freq[3]
    return float(obs_settl_rate), float(obs_ruin_rate)


def compute_gt_rates(gt):
    """Compute settlement and ruin rates from ground truth."""
    settl_mass = gt[:, :, 1].sum() + gt[:, :, 2].sum()
    ruin_mass = gt[:, :, 3].sum()
    total_mass = gt.sum()
    if total_mass < 1:
        return 0.0, 0.0
    return float(settl_mass / total_mass), float(ruin_mass / total_mass)


def grid_search_round(round_num, grid, settlements, gt, er_range, ws_range, n_sims, seed=42):
    """Run grid search for a single round."""
    best_score = -1.0
    best_er = float(er_range[0])
    best_ws = float(ws_range[0])
    all_results = []
    total = len(er_range) * len(ws_range)
    done = 0

    for er in er_range:
        for ws in ws_range:
            probs = simulate_monte_carlo(
                grid, settlements,
                n_sims=n_sims,
                expansion_rate=float(er),
                winter_severity=float(ws),
                seed=seed,
                n_workers=1,
            )
            score = compute_score(probs, gt)
            all_results.append((float(er), float(ws), score))
            done += 1

            if score > best_score:
                best_score = score
                best_er = float(er)
                best_ws = float(ws)

            if done % 10 == 0 or done == total:
                print(f"    [{done}/{total}] best: er={best_er:.3f} ws={best_ws:.3f} score={best_score:.2f}")

    return best_er, best_ws, best_score, all_results


def main():
    results = []
    overall_start = time.time()

    for round_num in [1, 2, 4, 5]:
        print(f"\n{'='*70}")
        print(f"ROUND {round_num}")
        print(f"{'='*70}")

        # Load data
        with open(os.path.join(DATA_DIR, f"round{round_num}_initial.json")) as f:
            r_data = json.load(f)

        grid = r_data["initial_states"][0]["grid"]
        settlements = r_data["initial_states"][0]["settlements"]
        gt = np.load(os.path.join(DATA_DIR, f"gt_r{round_num}_seed0.npy"))

        # Compute observation rates
        round_id = ROUNDS[round_num]["round_id"]
        obs_settl_rate, obs_ruin_rate = compute_obs_rates(round_id)
        gt_settl_rate, gt_ruin_rate = compute_gt_rates(gt)

        print(f"  Obs rates: settl={obs_settl_rate}, ruin={obs_ruin_rate}")
        print(f"  GT  rates: settl={gt_settl_rate:.4f}, ruin={gt_ruin_rate:.4f}")

        # Phase 1: Coarse grid search
        print(f"\n  Phase 1: Coarse search (step=0.03, n_sims=20)")
        t0 = time.time()
        er_coarse = np.arange(0.04, 0.19, 0.03)
        ws_coarse = np.arange(0.03, 0.21, 0.03)
        print(f"    Grid: {len(er_coarse)} x {len(ws_coarse)} = {len(er_coarse)*len(ws_coarse)} combos")

        best_er, best_ws, best_score, _ = grid_search_round(
            round_num, grid, settlements, gt,
            er_coarse, ws_coarse, n_sims=20, seed=42,
        )
        t1 = time.time()
        print(f"  Phase 1 done in {t1-t0:.0f}s: er={best_er:.3f} ws={best_ws:.3f} score={best_score:.2f}")

        # Phase 2: Fine grid search around best coarse result
        print(f"\n  Phase 2: Fine search (step=0.01, n_sims=30)")
        er_lo = max(0.02, best_er - 0.03)
        er_hi = min(0.22, best_er + 0.035)
        ws_lo = max(0.01, best_ws - 0.03)
        ws_hi = min(0.25, best_ws + 0.035)
        er_fine = np.arange(er_lo, er_hi, 0.01)
        ws_fine = np.arange(ws_lo, ws_hi, 0.01)
        print(f"    Grid: {len(er_fine)} x {len(ws_fine)} = {len(er_fine)*len(ws_fine)} combos")
        print(f"    ER range: [{er_lo:.3f}, {er_hi:.3f}), WS range: [{ws_lo:.3f}, {ws_hi:.3f})")

        best_er2, best_ws2, best_score2, _ = grid_search_round(
            round_num, grid, settlements, gt,
            er_fine, ws_fine, n_sims=30, seed=42,
        )
        t2 = time.time()
        print(f"  Phase 2 done in {t2-t1:.0f}s: er={best_er2:.3f} ws={best_ws2:.3f} score={best_score2:.2f}")

        # Use fine result if better
        if best_score2 >= best_score:
            final_er, final_ws, final_score = best_er2, best_ws2, best_score2
        else:
            final_er, final_ws, final_score = best_er, best_ws, best_score

        print(f"\n  >>> ROUND {round_num} BEST: er={final_er:.4f} ws={final_ws:.4f} score={final_score:.2f}")

        results.append({
            "round": round_num,
            "expansion_rate": round(final_er, 4),
            "winter_severity": round(final_ws, 4),
            "obs_settl_rate": round(obs_settl_rate, 4) if obs_settl_rate is not None else None,
            "obs_ruin_rate": round(obs_ruin_rate, 4) if obs_ruin_rate is not None else None,
            "gt_settl_rate": round(gt_settl_rate, 4),
            "gt_ruin_rate": round(gt_ruin_rate, 4),
            "score": round(final_score, 2),
        })

    # Save calibration data
    cal_data = {"rounds": results}
    cal_path = os.path.join(DATA_DIR, "sim_calibration.json")
    with open(cal_path, "w") as f:
        json.dump(cal_data, f, indent=2)
    print(f"\nCalibration saved to {cal_path}")

    # Summary table
    total_time = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"CALIBRATION SUMMARY (total time: {total_time:.0f}s)")
    print(f"{'='*70}")
    print(f"{'Round':>5} {'ER':>8} {'WS':>8} {'Score':>7} {'ObsSettl':>10} {'ObsRuin':>10} {'GTSettl':>10} {'GTRuin':>10}")
    print("-" * 70)
    for r in results:
        obs_s = f"{r['obs_settl_rate']:.4f}" if r['obs_settl_rate'] is not None else "N/A"
        obs_r = f"{r['obs_ruin_rate']:.4f}" if r['obs_ruin_rate'] is not None else "N/A"
        print(f"{r['round']:>5} {r['expansion_rate']:>8.4f} {r['winter_severity']:>8.4f} "
              f"{r['score']:>7.2f} {obs_s:>10} {obs_r:>10} "
              f"{r['gt_settl_rate']:>10.4f} {r['gt_ruin_rate']:>10.4f}")

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\nAverage score across rounds: {avg_score:.2f}")


if __name__ == "__main__":
    main()
