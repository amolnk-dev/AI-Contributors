#!/usr/bin/env python3
"""Generate visualization data for the Astar Island prediction inspector.

Loads initial states, ground truth, and builds predictions using the current model.
Computes per-cell KL divergence and exports everything as data.json for viz.html.

Usage:
    cd tasks/astar-island && python viz/generate_viz_data.py
    cd tasks/astar-island/viz && python generate_viz_data.py

Then open viz.html (via python -m http.server in the viz directory).
"""

import json
import os
import sys

# Allow importing from the parent directory (tasks/astar-island/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from model import build_prediction, load_calibration
from utils import normalize_prediction, load_observations
from evaluate import compute_score, kl_divergence, entropy
from dtos import CLASS_NAMES, NUM_CLASSES, TERRAIN_TO_CLASS

# Round number -> round UUID mapping
ROUND_IDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    3: "c5cdf100-a876-4fb7-b5d8-757162c97989",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    6: "ae78003a-4efe-425a-881a-d16a39bca0ad",
    7: "36e581f1-73f8-453f-ab98-cbe3052b701b",
}

# Resolve DATA_DIR: prefer relative to script location, fall back to cwd-based path.
# This handles both `cd tasks/astar-island && python viz/generate_viz_data.py`
# and `cd tasks/astar-island/viz && python generate_viz_data.py`.
_script_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_cwd_data = os.path.join(os.getcwd(), "data")

if os.path.isdir(_script_data) and os.path.exists(os.path.join(_script_data, "calibration.json")):
    DATA_DIR = _script_data
elif os.path.isdir(_cwd_data):
    DATA_DIR = _cwd_data
else:
    DATA_DIR = _script_data  # fallback, will report missing files


def load_initial_state(round_num: int) -> dict:
    """Load round initial state JSON."""
    path = os.path.join(DATA_DIR, f"round{round_num}_initial.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_ground_truth(round_num: int, seed: int) -> np.ndarray | None:
    """Load ground truth probability tensor (H x W x 6)."""
    path = os.path.join(DATA_DIR, f"gt_r{round_num}_seed{seed}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def process_round(round_num: int, calibration: dict | None) -> dict | None:
    """Process one round: build predictions, compute errors, return viz data."""
    initial = load_initial_state(round_num)
    if initial is None:
        print(f"  Round {round_num}: no initial state file, skipping")
        return None

    round_id = ROUND_IDS.get(round_num)
    if round_id is None:
        print(f"  Round {round_num}: no round ID mapping, skipping")
        return None

    # Load observations for this round.
    # Try load_observations() first (uses utils.py internal path),
    # then fall back to loading directly from DATA_DIR.
    observations = load_observations(round_id)
    if not observations:
        obs_path = os.path.join(DATA_DIR, f"obs_{round_id}.jsonl")
        if os.path.exists(obs_path):
            observations = []
            with open(obs_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        observations.append(json.loads(line))

    seeds_count = initial.get("seeds_count", len(initial.get("initial_states", [])))
    all_grids = [initial["initial_states"][i]["grid"] for i in range(seeds_count)]

    round_data = {
        "round_num": round_num,
        "round_id": round_id,
        "seeds": [],
    }

    round_scores = []

    for seed_idx in range(seeds_count):
        gt = load_ground_truth(round_num, seed_idx)
        if gt is None:
            print(f"  Round {round_num}, seed {seed_idx}: no ground truth, skipping")
            continue

        grid = all_grids[seed_idx]
        settlements_raw = initial["initial_states"][seed_idx].get("settlements", [])
        # Ensure settlements are dicts
        settlements = []
        for s in settlements_raw:
            if isinstance(s, dict):
                settlements.append(s)
            else:
                settlements.append({"x": s.get("x", 0), "y": s.get("y", 0)})

        same_seed_obs = [o for o in observations if o.get("seed_index") == seed_idx]

        # Build prediction
        prediction = build_prediction(
            initial_grid=grid,
            settlements=settlements,
            observations=same_seed_obs,
            seed_index=seed_idx,
            all_initial_grids=all_grids,
            all_observations=same_seed_obs,
            calibration=calibration,
        )
        prediction = normalize_prediction(prediction)

        # Compute metrics
        kl = kl_divergence(gt, prediction)  # H x W
        ent = entropy(gt)  # H x W
        score = compute_score(prediction, gt)
        round_scores.append(score)

        # Convert initial grid to class indices
        h, w = len(grid), len(grid[0])
        initial_classes = [[TERRAIN_TO_CLASS.get(grid[y][x], 0) for x in range(w)] for y in range(h)]

        # Argmax of prediction and ground truth
        pred_argmax = prediction.argmax(axis=-1).tolist()
        gt_argmax = gt.argmax(axis=-1).tolist()

        # Per-class error breakdown: average KL for cells where GT argmax is each class
        per_class_error = {}
        for cls_idx in range(NUM_CLASSES):
            mask = gt.argmax(axis=-1) == cls_idx
            if mask.sum() > 0:
                per_class_error[CLASS_NAMES[cls_idx]] = {
                    "mean_kl": float(kl[mask].mean()),
                    "max_kl": float(kl[mask].max()),
                    "count": int(mask.sum()),
                }
            else:
                per_class_error[CLASS_NAMES[cls_idx]] = {
                    "mean_kl": 0.0,
                    "max_kl": 0.0,
                    "count": 0,
                }

        seed_data = {
            "seed": seed_idx,
            "score": round(score, 4),
            "initial_grid": initial_classes,
            "pred_argmax": pred_argmax,
            "gt_argmax": gt_argmax,
            "kl_map": [[round(float(kl[y][x]), 6) for x in range(w)] for y in range(h)],
            "entropy_map": [[round(float(ent[y][x]), 6) for x in range(w)] for y in range(h)],
            # Full probability vectors (6 floats per cell) for tooltip
            "pred_probs": [
                [[round(float(prediction[y, x, c]), 4) for c in range(NUM_CLASSES)] for x in range(w)]
                for y in range(h)
            ],
            "gt_probs": [
                [[round(float(gt[y, x, c]), 4) for c in range(NUM_CLASSES)] for x in range(w)]
                for y in range(h)
            ],
            "per_class_error": per_class_error,
        }
        round_data["seeds"].append(seed_data)
        print(f"  Round {round_num}, seed {seed_idx}: score={score:.2f}")

    if not round_data["seeds"]:
        return None

    round_data["avg_score"] = round(sum(round_scores) / len(round_scores), 4) if round_scores else 0.0
    return round_data


def main():
    print("Generating visualization data...")
    print(f"Data directory: {os.path.abspath(DATA_DIR)}")

    calibration = load_calibration()
    if not calibration:
        # Fallback: load directly from DATA_DIR
        cal_path = os.path.join(DATA_DIR, "calibration.json")
        if os.path.exists(cal_path):
            with open(cal_path) as f:
                calibration = json.load(f)
    if calibration:
        print("Loaded calibration data")
    else:
        print("No calibration data found")

    viz_data = {
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "rounds": [],
    }

    for round_num in sorted(ROUND_IDS.keys()):
        print(f"\nProcessing round {round_num}...")
        round_data = process_round(round_num, calibration)
        if round_data:
            viz_data["rounds"].append(round_data)

    # Write output to the same directory as this script (viz/)
    viz_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(viz_dir, "data.json")
    with open(out_path, "w") as f:
        json.dump(viz_data, f)

    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nWrote {out_path} ({file_size_mb:.1f} MB)")
    print(f"Rounds: {len(viz_data['rounds'])}")
    for rd in viz_data["rounds"]:
        print(f"  Round {rd['round_num']}: {len(rd['seeds'])} seeds, avg_score={rd['avg_score']}")


if __name__ == "__main__":
    main()
