"""Astar Island runner — main entry point.

Fetches round details, strategically queries the simulator, builds
predictions using the multi-layer heuristic model, and submits.

Usage:
  python run.py                    # Run with active round
  python run.py --round-id UUID    # Run for specific round
  python run.py --dry-run          # Build predictions without submitting
  python run.py --resume           # Resume from checkpoint

Flow:
  1. Fetch active round + initial states (free)
  2. Analyze initial grids → identify dynamic cells
  3. Compute viewport placements → greedy set cover
  4. Allocate query budget (50 total across 5 seeds)
  5. Execute queries (with checkpoint persistence)
  6. Build prediction tensors (multi-layer heuristic)
  7. Normalize + validate
  8. Submit all 5 seeds
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from client import AstarClient, BudgetExhaustedError, AuthExpiredError
from model import build_prediction, load_calibration, ensemble_predictions
from utils import (
    identify_dynamic_cells,
    compute_viewport_placements,
    allocate_queries,
    normalize_prediction,
    save_observation,
    load_observations,
    plot_prediction_heatmap,
    plot_initial_grid,
)

logger = logging.getLogger(__name__)


def find_active_round(client: AstarClient) -> dict | None:
    """Find the currently active round."""
    rounds = client.get_rounds()
    for r in rounds:
        if r.status == "active":
            return r
    return None


def execute_queries(
    client: AstarClient,
    round_id: str,
    query_plan: list[list[tuple]],
    existing_observations: list[dict],
) -> list[dict]:
    """Execute the query plan, persisting observations after each call.

    Resumes from checkpoint if observations already exist.
    """
    all_observations = list(existing_observations)

    # Count existing queries per seed+viewport to avoid re-querying
    existing_keys = set()
    for obs in existing_observations:
        vp = obs["viewport"]
        key = (obs["seed_index"], vp["x"], vp["y"], vp["w"], vp["h"])
        existing_keys.add(key)

    # Track how many times we've observed each viewport per seed
    vp_counts: dict[tuple, int] = {}
    for obs in existing_observations:
        vp = obs["viewport"]
        key = (obs["seed_index"], vp["x"], vp["y"])
        vp_counts[key] = vp_counts.get(key, 0) + 1

    for seed_idx, seed_plan in enumerate(query_plan):
        for viewport, num_repeats in seed_plan:
            vp_key = (seed_idx, viewport.x, viewport.y)
            already_done = vp_counts.get(vp_key, 0)
            remaining = max(0, num_repeats - already_done)

            for rep in range(remaining):
                try:
                    result = client.simulate(
                        round_id=round_id,
                        seed_index=seed_idx,
                        viewport_x=viewport.x,
                        viewport_y=viewport.y,
                        viewport_w=viewport.w,
                        viewport_h=viewport.h,
                    )
                except BudgetExhaustedError:
                    logger.warning("Budget exhausted! Stopping queries.")
                    return all_observations
                except AuthExpiredError:
                    logger.error(
                        "AUTH TOKEN EXPIRED! Refresh your token from app.ainm.no "
                        "and update ASTAR_TOKEN, then re-run with --resume"
                    )
                    return all_observations

                # Build observation record
                obs = {
                    "seed_index": seed_idx,
                    "viewport": {
                        "x": result.viewport.x,
                        "y": result.viewport.y,
                        "w": result.viewport.w,
                        "h": result.viewport.h,
                    },
                    "grid": result.grid,
                    "settlements": [s.model_dump() for s in result.settlements],
                    "queries_used": result.queries_used,
                    "queries_max": result.queries_max,
                }

                # Persist immediately
                save_observation(round_id, obs)
                all_observations.append(obs)

                logger.info(
                    f"  Query {result.queries_used}/{result.queries_max}: "
                    f"seed={seed_idx} vp=({viewport.x},{viewport.y}) "
                    f"repeat={already_done + rep + 1}/{num_repeats}"
                )

    return all_observations


def build_all_predictions(
    round_info,
    all_observations: list[dict],
    calibration: dict | None,
    use_ensemble: bool = False,
) -> list[np.ndarray]:
    """Build prediction tensors for all seeds."""
    num_seeds = round_info.seeds_count
    all_initial_grids = [
        round_info.initial_states[i].grid for i in range(num_seeds)
    ]

    predictions = []
    for seed_idx in range(num_seeds):
        initial_grid = all_initial_grids[seed_idx]
        settlements = [
            s.model_dump()
            for s in round_info.initial_states[seed_idx].settlements
        ]

        # Same-seed observations for Layer 2 (disabled) and seed-specific logic
        same_seed_obs = [
            o for o in all_observations if o["seed_index"] == seed_idx
        ]

        # All settlements for empirical tables
        all_settl = [
            [s.model_dump() for s in round_info.initial_states[i].settlements]
            for i in range(num_seeds)
        ]

        # TODO: Look at using all 50 observations more effectively.
        # Previously we only passed same-seed obs to all_observations,
        # which starved L7/empirical tables of cross-seed data.
        # On extinction rounds like R10 this cost us ~13 points (75→88).
        # Now we pass ALL observations so L7 can detect round-wide patterns.
        prediction = build_prediction(
            initial_grid=initial_grid,
            settlements=settlements,
            observations=same_seed_obs,
            seed_index=seed_idx,
            all_initial_grids=all_initial_grids,
            all_observations=all_observations,  # ALL seeds for L7/empirical tables
            calibration=calibration,
            all_settlements=all_settl,
        )

        # Final safety normalization
        prediction = normalize_prediction(prediction)
        predictions.append(prediction)

        logger.info(
            f"Seed {seed_idx}: prediction shape={prediction.shape}, "
            f"min={prediction.min():.4f}, max={prediction.max():.4f}"
        )

    return predictions


def submit_all(
    client: AstarClient,
    round_id: str,
    predictions: list[np.ndarray],
):
    """Submit prediction tensors for all seeds."""
    for seed_idx, prediction in enumerate(predictions):
        try:
            result = client.submit(
                round_id=round_id,
                seed_index=seed_idx,
                prediction=prediction.tolist(),
            )
            logger.info(f"Submitted seed {seed_idx}: {result}")
        except Exception as e:
            logger.error(f"Failed to submit seed {seed_idx}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Astar Island prediction runner")
    parser.add_argument("--round-id", help="Specific round ID (default: active round)")
    parser.add_argument("--dry-run", action="store_true", help="Build predictions without submitting")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--no-ensemble", action="store_true", help="Skip ensemble, use single model")
    parser.add_argument("--visualize", action="store_true", help="Save prediction visualizations")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Initialize client
    try:
        client = AstarClient()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Find round
    if args.round_id:
        round_id = args.round_id
    else:
        active = find_active_round(client)
        if active is None:
            logger.error("No active round found. Check https://app.ainm.no")
            sys.exit(1)
        round_id = active.id
        logger.info(f"Active round: {active.round_number} ({round_id})")

    # Fetch round details
    round_info = client.get_round_detail(round_id)
    logger.info(
        f"Round: {round_info.map_width}x{round_info.map_height}, "
        f"{round_info.seeds_count} seeds"
    )

    # Check budget
    budget = client.get_budget()
    logger.info(f"Budget: {budget.queries_used}/{budget.queries_max} used")

    # Load existing observations (checkpoint)
    existing_obs = load_observations(round_id)
    if existing_obs:
        logger.info(f"Loaded {len(existing_obs)} observations from checkpoint")

    # Visualize initial grids
    if args.visualize:
        for i in range(round_info.seeds_count):
            plot_initial_grid(
                round_info.initial_states[i].grid,
                title=f"Seed {i} Initial",
            )

    # Analyze initial states and compute query plan
    logger.info("Computing query plan...")
    viewport_placements = []
    for seed_idx in range(round_info.seeds_count):
        state = round_info.initial_states[seed_idx]
        settlements = [s.model_dump() for s in state.settlements]
        dynamic = identify_dynamic_cells(state.grid, settlements)
        placements = compute_viewport_placements(
            dynamic,
            map_width=round_info.map_width,
            map_height=round_info.map_height,
        )
        viewport_placements.append(placements)
        logger.info(
            f"  Seed {seed_idx}: {len(dynamic)} dynamic cells, "
            f"{len(placements)} viewports needed"
        )

    remaining_budget = budget.queries_max - budget.queries_used
    query_plan = allocate_queries(
        num_seeds=round_info.seeds_count,
        viewport_placements=viewport_placements,
        total_budget=remaining_budget,
    )

    total_planned = sum(
        repeats for seed_plan in query_plan for _, repeats in seed_plan
    )
    logger.info(f"Query plan: {total_planned} queries across {round_info.seeds_count} seeds")

    # Execute queries
    if not args.resume or not existing_obs:
        logger.info("Executing queries...")
        all_observations = execute_queries(
            client, round_id, query_plan, existing_obs,
        )
    else:
        logger.info("Resuming from checkpoint, skipping queries")
        all_observations = existing_obs

    logger.info(f"Total observations: {len(all_observations)}")

    # Load calibration from past rounds
    calibration = load_calibration()
    if calibration:
        logger.info(
            f"Loaded calibration from {calibration.get('rounds_analyzed', 0)} past rounds"
        )

    # Build predictions
    logger.info("Building predictions...")
    predictions = build_all_predictions(
        round_info,
        all_observations,
        calibration,
        use_ensemble=not args.no_ensemble,
    )

    # Visualize predictions
    if args.visualize:
        for seed_idx, pred in enumerate(predictions):
            plot_prediction_heatmap(
                pred,
                initial_grid=round_info.initial_states[seed_idx].grid,
                title=f"Seed {seed_idx} Prediction",
            )

    # Submit first prediction (safe)
    if args.dry_run:
        logger.info("DRY RUN — not submitting predictions")
    else:
        logger.info("Submitting predictions (pass 1 - safe)...")
        submit_all(client, round_id, predictions)
        logger.info("Safe prediction submitted!")

    # ── PHASE 2+3: Use remaining budget for refinement ──
    budget = client.get_budget()
    remaining = budget.queries_max - budget.queries_used
    if remaining > 0 and not args.dry_run:
        logger.info(f"")
        logger.info(f"{'='*50}")
        logger.info(f"  REFINEMENT PHASE: {remaining} queries remaining")
        logger.info(f"{'='*50}")

        # Use depth strategy for remaining queries (repeat top viewports)
        refine_plan = allocate_queries(
            num_seeds=round_info.seeds_count,
            viewport_placements=viewport_placements,
            total_budget=remaining,
            repeat_strategy="depth",
        )
        all_observations = execute_queries(client, round_id, refine_plan, all_observations)
        logger.info(f"Total observations after refinement: {len(all_observations)}")

        # Rebuild predictions with all data (cross-seed tables now have more data)
        logger.info("Rebuilding predictions with full data...")
        predictions = build_all_predictions(
            round_info, all_observations, calibration,
            use_ensemble=not args.no_ensemble,
        )

        # Resubmit improved predictions
        logger.info("RESUBMITTING improved predictions (pass 2)...")
        submit_all(client, round_id, predictions)
        logger.info("Improved prediction submitted!")

    logger.info(f"All phases complete! Total observations: {len(all_observations)}")


if __name__ == "__main__":
    main()
