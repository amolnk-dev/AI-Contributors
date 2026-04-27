"""Post-round analysis — fetch ground truth and extract calibration parameters.

Run after a round completes to learn from the gap between predictions and reality.
Produces data/calibration.json which model.py loads for future rounds.
"""

import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from client import AstarClient
from dtos import TERRAIN_TO_CLASS, NUM_CLASSES, STATIC_TERRAIN_CODES
from utils import grid_to_class_array

logger = logging.getLogger(__name__)


def _is_coastal(grid: list[list[int]], y: int, x: int) -> bool:
    h, w = len(grid), len(grid[0])
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and grid[ny][nx] == 10:
                return True
    return False


def _count_adjacent_forests(grid: list[list[int]], y: int, x: int) -> int:
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


def analyze_round(round_id: str, client: AstarClient) -> dict:
    """Analyze a completed round and extract calibration parameters.

    Fetches ground truth for all 5 seeds and computes:
    - Per-context transition probabilities
    - Optimal blend weights
    """
    logger.info(f"Analyzing round {round_id}...")

    # Fetch round details for initial grids
    round_info = client.get_round_detail(round_id)
    num_seeds = round_info.seeds_count

    # Collect ground truth distributions per context
    # context = "terrainCode_coastalBool_forestsCount"
    context_distributions: dict[str, list[np.ndarray]] = {}
    total_cells = 0
    total_score = 0.0

    for seed_idx in range(num_seeds):
        try:
            analysis = client.get_analysis(round_id, seed_idx)
        except Exception as e:
            logger.warning(f"Could not fetch analysis for seed {seed_idx}: {e}")
            continue

        gt = np.array(analysis.ground_truth)
        initial_grid = round_info.initial_states[seed_idx].grid
        h, w = len(initial_grid), len(initial_grid[0])

        if analysis.score is not None:
            total_score += analysis.score

        for y in range(h):
            for x in range(w):
                code = initial_grid[y][x]
                if code in STATIC_TERRAIN_CODES:
                    continue

                coastal = _is_coastal(initial_grid, y, x)
                adj_forests = min(_count_adjacent_forests(initial_grid, y, x), 3)
                context = f"{code}_coastal{int(coastal)}_forests{adj_forests}"

                if context not in context_distributions:
                    context_distributions[context] = []
                context_distributions[context].append(gt[y, x])
                total_cells += 1

    # Average distributions per context
    context_priors = {}
    for context, dists in context_distributions.items():
        avg = np.mean(dists, axis=0)
        context_priors[context] = avg.tolist()

    calibration = {
        "round_id": round_id,
        "num_seeds": num_seeds,
        "total_dynamic_cells": total_cells,
        "avg_score": total_score / num_seeds if num_seeds > 0 else 0,
        "blend_weight": 0.5,  # Can be tuned
        "context_priors": context_priors,
    }

    # Save calibration
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    cal_path = os.path.join(data_dir, "calibration.json")

    # Merge with existing calibration if present
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            existing = json.load(f)
        # Average new priors with existing ones
        existing_priors = existing.get("context_priors", {})
        for context, new_prior in context_priors.items():
            if context in existing_priors:
                old = np.array(existing_priors[context])
                new = np.array(new_prior)
                merged = (old + new) / 2
                context_priors[context] = merged.tolist()
        calibration["context_priors"] = {**existing_priors, **context_priors}
        calibration["rounds_analyzed"] = existing.get("rounds_analyzed", 0) + 1
    else:
        calibration["rounds_analyzed"] = 1

    with open(cal_path, "w") as f:
        json.dump(calibration, f, indent=2)

    logger.info(
        f"Calibration saved: {len(context_priors)} contexts, "
        f"avg score {calibration['avg_score']:.1f}"
    )
    return calibration


def main():
    """CLI: python analyze.py [round_id]"""
    logging.basicConfig(level=logging.INFO)
    client = AstarClient()

    if len(sys.argv) > 1:
        round_id = sys.argv[1]
    else:
        # Find most recent completed round
        rounds = client.get_my_rounds()
        completed = [r for r in rounds if r["status"] in ("completed", "scoring")]
        if not completed:
            print("No completed rounds to analyze.")
            return
        round_id = completed[-1]["id"]
        print(f"Analyzing most recent completed round: {round_id}")

    calibration = analyze_round(round_id, client)
    print(f"\nCalibration extracted:")
    print(f"  Contexts: {len(calibration['context_priors'])}")
    print(f"  Avg score: {calibration['avg_score']:.1f}")
    print(f"  Rounds analyzed: {calibration['rounds_analyzed']}")


if __name__ == "__main__":
    main()
