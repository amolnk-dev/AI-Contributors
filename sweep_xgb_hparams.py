"""Sweep XGBoost hyperparameters per terrain type in 5-fold LORO.

Sweeps one terrain at a time (holding others at defaults).
Usage:
    python sweep_xgb_hparams.py plains
    python sweep_xgb_hparams.py forest
    python sweep_xgb_hparams.py settl
    python sweep_xgb_hparams.py all
"""

import itertools
import sys
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from sweep_l7_class import evaluate_loro, BASE_L7

# Sweep grids per terrain
SWEEP_GRIDS = {
    "plains": {
        "max_depth": [4, 5, 6],
        "n_estimators": [200, 300],
        "min_child_weight": [2, 3, 5],
    },
    "forest": {
        "max_depth": [3, 4, 5],
        "n_estimators": [150, 200, 300],
    },
    "settl": {
        "max_depth": [2, 3, 4],
        "n_estimators": [100, 150, 200],
        "min_child_weight": [5, 8],
    },
}

BLEND = 0.35


def generate_configs(terrain):
    """Generate all hparam combinations for a terrain."""
    grid = SWEEP_GRIDS[terrain]
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    configs = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


def sweep_terrain(terrain):
    """Sweep hparams for one terrain, report LORO scores."""
    configs = generate_configs(terrain)
    print(f"\n{'='*60}")
    print(f"Sweeping {terrain}: {len(configs)} configs")
    print(f"{'='*60}\n")

    best_avg = -1
    best_config = None

    for i, config in enumerate(configs):
        xgb_hparams = {terrain: config}
        avg, per_round = evaluate_loro(BASE_L7, BLEND, include_r3=False, xgb_hparams=xgb_hparams)
        rounds_str = " ".join(f"R{r}={s:.2f}" for r, s in sorted(per_round.items()))
        config_str = " ".join(f"{k}={v}" for k, v in sorted(config.items()))
        print(f"  [{i+1}/{len(configs)}] {config_str}: avg={avg:.2f}  {rounds_str}")

        if avg > best_avg:
            best_avg = avg
            best_config = config

    print(f"\nBEST {terrain}: {best_config} -> avg={best_avg:.2f}")
    return best_config, best_avg


def main():
    if len(sys.argv) < 2:
        print("Usage: python sweep_xgb_hparams.py <plains|forest|settl|all>")
        return

    terrain = sys.argv[1].lower()

    if terrain == "all":
        results = {}
        for t in ["plains", "forest", "settl"]:
            config, avg = sweep_terrain(t)
            results[t] = (config, avg)

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for t, (config, avg) in results.items():
            print(f"  {t}: avg={avg:.2f} {config}")

        # Run combined eval with all best hparams
        combined_hparams = {t: config for t, (config, _) in results.items()}
        print(f"\nRunning combined eval with all best hparams...")
        avg, per_round = evaluate_loro(BASE_L7, BLEND, include_r3=False, xgb_hparams=combined_hparams)
        rounds_str = " ".join(f"R{r}={s:.2f}" for r, s in sorted(per_round.items()))
        print(f"COMBINED: avg={avg:.2f}  {rounds_str}")
    elif terrain in SWEEP_GRIDS:
        sweep_terrain(terrain)
    else:
        print(f"Unknown terrain: {terrain}. Use plains, forest, settl, or all.")


if __name__ == "__main__":
    main()
