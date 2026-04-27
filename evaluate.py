"""Offline evaluation — scores predictions against ground truth.

This is the autoresearch-compatible evaluation script.
It rebuilds predictions from saved observations and scores them
against ground truth from completed rounds.

Usage:
  python evaluate.py                    # Score latest completed round
  python evaluate.py --round-id UUID    # Score specific round

Prints: val_metric: <float>  (autoresearch reads this line)

The score formula matches the competition's entropy-weighted KL divergence:
  score = max(0, min(100, 100 * exp(-3 * weighted_kl)))
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from client import AstarClient
from model import build_prediction, load_calibration, ensemble_predictions
from utils import normalize_prediction, load_observations

logger = logging.getLogger(__name__)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-cell KL divergence: KL(p || q) = Σ pᵢ × log(pᵢ / qᵢ).

    p = ground truth, q = our prediction.
    Returns H×W array of KL values.
    """
    # Avoid log(0) by clamping
    p_safe = np.maximum(p, 1e-10)
    q_safe = np.maximum(q, 1e-10)
    kl = np.sum(p_safe * np.log(p_safe / q_safe), axis=-1)
    return kl


def entropy(p: np.ndarray) -> np.ndarray:
    """Per-cell entropy: H(p) = -Σ pᵢ × log(pᵢ).

    Returns H×W array. Higher entropy = more uncertain = more important for scoring.
    """
    p_safe = np.maximum(p, 1e-10)
    return -np.sum(p_safe * np.log(p_safe), axis=-1)


def compute_score(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute the competition score for one seed.

    score = max(0, min(100, 100 × exp(-3 × weighted_kl)))

    where weighted_kl = Σ entropy(cell) × KL(truth, pred) / Σ entropy(cell)
    Only dynamic cells (entropy > 0) contribute.
    """
    kl = kl_divergence(ground_truth, prediction)
    ent = entropy(ground_truth)

    # Only score cells with non-trivial entropy
    total_entropy = ent.sum()
    if total_entropy < 1e-10:
        return 100.0  # All static — perfect by default

    weighted_kl = (ent * kl).sum() / total_entropy
    score = max(0.0, min(100.0, 100.0 * np.exp(-3.0 * weighted_kl)))
    return score


def evaluate_round(round_id: str, client: AstarClient) -> float:
    """Evaluate our model against ground truth for a completed round.

    Returns the average score across all seeds.
    """
    round_info = client.get_round_detail(round_id)
    all_observations = load_observations(round_id)
    calibration = load_calibration()
    all_initial_grids = [
        round_info.initial_states[i].grid
        for i in range(round_info.seeds_count)
    ]

    scores = []
    for seed_idx in range(round_info.seeds_count):
        # Fetch ground truth
        try:
            analysis = client.get_analysis(round_id, seed_idx)
        except Exception as e:
            logger.warning(f"Seed {seed_idx}: could not fetch ground truth: {e}")
            continue

        ground_truth = np.array(analysis.ground_truth)

        # Build our prediction (same pipeline as run.py)
        initial_grid = all_initial_grids[seed_idx]
        settlements = [
            s.model_dump()
            for s in round_info.initial_states[seed_idx].settlements
        ]

        # Use same-seed observations only (cross-seed transfer disabled)
        same_seed_obs = [
            o for o in all_observations if o["seed_index"] == seed_idx
        ]
        prediction = build_prediction(
            initial_grid=initial_grid,
            settlements=settlements,
            observations=same_seed_obs,
            seed_index=seed_idx,
            all_initial_grids=all_initial_grids,
            all_observations=same_seed_obs,
            calibration=calibration,
        )
        prediction = normalize_prediction(prediction)

        # Score
        seed_score = compute_score(prediction, ground_truth)
        scores.append(seed_score)
        logger.info(f"Seed {seed_idx}: score={seed_score:.2f}")

    if not scores:
        logger.error("No seeds could be scored")
        return 0.0

    avg_score = sum(scores) / len(scores)
    return avg_score


def main():
    parser = argparse.ArgumentParser(description="Offline evaluation against ground truth")
    parser.add_argument("--round-id", help="Specific round ID (default: latest completed)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    client = AstarClient()

    if args.round_id:
        round_id = args.round_id
    else:
        rounds = client.get_my_rounds()
        completed = [
            r for r in rounds if r["status"] in ("completed", "scoring")
        ]
        if not completed:
            print("No completed rounds to evaluate against.")
            print("val_metric: 0.0")
            return
        round_id = completed[-1]["id"]
        logger.info(f"Evaluating against round: {round_id}")

    score = evaluate_round(round_id, client)

    # Autoresearch reads this line
    print(f"val_metric: {score:.4f}")


if __name__ == "__main__":
    main()
