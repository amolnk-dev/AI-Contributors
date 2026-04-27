"""Autoresearch evaluation harness — READ ONLY, DO NOT MODIFY.

This file wraps the evaluation pipeline for the autoresearch protocol.
It loads observations from Round 1, builds predictions using the current
model.py, scores against ground truth, and prints val_metric.

Usage:
    cd tasks/astar-island && python prepare.py > run.log 2>&1
    grep "^val_metric:" run.log

Contract:
    - TIME_BUDGET: not applicable (eval takes <2 seconds)
    - evaluate(): builds predictions and scores against R1 ground truth
    - Prints val_metric: <float> (higher is better, max 100)
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

TIME_BUDGET = 30  # seconds (eval takes <2s, generous margin)


def evaluate():
    """Build predictions with current model.py and score against R1 ground truth."""
    from client import AstarClient
    from model import build_prediction, load_calibration
    from utils import normalize_prediction, load_observations
    from evaluate import compute_score

    token = os.environ.get("ASTAR_TOKEN", "")
    if not token:
        print("ERROR: Set ASTAR_TOKEN env var")
        return 0.0

    client = AstarClient(token=token)

    # Use Round 1 ground truth for offline evaluation
    round_id = "71451d74-be9f-471f-aacd-a41f3b68a9cd"

    try:
        detail = client.get_round_detail(round_id)
    except Exception as e:
        print(f"ERROR: Could not fetch round details: {e}")
        return 0.0

    all_obs = load_observations(round_id)
    if not all_obs:
        print("ERROR: No observations found for Round 1. Run round 1 first.")
        return 0.0

    all_grids = [detail.initial_states[i].grid for i in range(detail.seeds_count)]
    cal = load_calibration()

    scores = []
    for seed_idx in range(detail.seeds_count):
        try:
            analysis = client.get_analysis(round_id, seed_idx)
        except Exception as e:
            print(f"Seed {seed_idx}: could not fetch ground truth: {e}")
            continue

        ground_truth = np.array(analysis.ground_truth)
        initial_grid = all_grids[seed_idx]
        settlements = [s.model_dump() for s in detail.initial_states[seed_idx].settlements]
        same_seed_obs = [o for o in all_obs if o["seed_index"] == seed_idx]

        prediction = build_prediction(
            initial_grid=initial_grid,
            settlements=settlements,
            observations=same_seed_obs,
            seed_index=seed_idx,
            all_initial_grids=all_grids,
            all_observations=same_seed_obs,
            calibration=cal,
        )
        prediction = normalize_prediction(prediction)

        seed_score = compute_score(prediction, ground_truth)
        scores.append(seed_score)
        print(f"seed_{seed_idx}_score: {seed_score:.4f}")

    if not scores:
        return 0.0

    avg = sum(scores) / len(scores)
    return avg


if __name__ == "__main__":
    start = time.time()
    score = evaluate()
    elapsed = time.time() - start

    print("---")
    print(f"val_metric: {score:.6f}")
    print(f"total_seconds: {elapsed:.1f}")
    print(f"peak_vram_mb: 0.0")
