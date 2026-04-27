"""Astar Island utilities — grid analysis, viewport strategy, normalization, visualization."""

import json
import os
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from dtos import (
    TERRAIN_TO_CLASS,
    NUM_CLASSES,
    CLASS_NAMES,
    STATIC_TERRAIN_CODES,
    PROB_FLOOR,
    InitialState,
    Viewport,
)


# ──────────────────────────────────────────────────────────────
# Grid parsing
# ──────────────────────────────────────────────────────────────

def grid_to_class_array(grid: list[list[int]]) -> np.ndarray:
    """Convert a terrain code grid to class indices (H×W)."""
    return np.vectorize(lambda code: TERRAIN_TO_CLASS.get(code, 0))(
        np.array(grid, dtype=np.int32)
    )


def identify_dynamic_cells(
    grid: list[list[int]],
    settlements: list[dict] | None = None,
    radius: int = 3,
) -> set[tuple[int, int]]:
    """Identify cells that may change during simulation.

    Dynamic cells are:
    1. Cells with dynamic terrain codes (settlement, port, ruin, forest)
    2. Cells within `radius` of any settlement (expansion zone)

    Returns set of (y, x) coordinates.
    """
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    dynamic = set()

    # Mark cells with dynamic terrain
    for y in range(h):
        for x in range(w):
            if grid[y][x] not in STATIC_TERRAIN_CODES:
                dynamic.add((y, x))

    # Expand around settlements
    if settlements:
        for s in settlements:
            sx, sy = s.get("x", s.get("sx", 0)), s.get("y", s.get("sy", 0))
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = sy + dy, sx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        # Only mark non-ocean, non-mountain cells as dynamic
                        if grid[ny][nx] not in {10, 5}:
                            dynamic.add((ny, nx))

    return dynamic


# ──────────────────────────────────────────────────────────────
# Viewport strategy — greedy set cover of dynamic cells
# ──────────────────────────────────────────────────────────────

def compute_viewport_placements(
    dynamic_cells: set[tuple[int, int]],
    map_width: int = 40,
    map_height: int = 40,
    viewport_w: int = 15,
    viewport_h: int = 15,
) -> list[Viewport]:
    """Greedy set-cover: place viewports to maximize coverage of dynamic cells.

    ┌───────────────────────────────────────────┐
    │  40×40 map                                 │
    │                                            │
    │   ┌──────────┐                             │
    │   │ viewport1│    ┌──────────┐             │
    │   │ (15×15)  │    │ viewport2│             │
    │   │ covers   │    │          │             │
    │   │ cluster1 │    │ cluster2 │             │
    │   └──────────┘    └──────────┘             │
    │                                            │
    │              ┌──────────┐                  │
    │              │ viewport3│                  │
    │              │          │                  │
    │              └──────────┘                  │
    └───────────────────────────────────────────┘
    """
    if not dynamic_cells:
        return []

    uncovered = set(dynamic_cells)
    placements = []

    while uncovered:
        best_vp = None
        best_count = 0

        # Try all possible viewport positions (step by 1 for precision)
        for vy in range(0, map_height - viewport_h + 1, 1):
            for vx in range(0, map_width - viewport_w + 1, 1):
                covered = 0
                for cy, cx in uncovered:
                    if vy <= cy < vy + viewport_h and vx <= cx < vx + viewport_w:
                        covered += 1
                if covered > best_count:
                    best_count = covered
                    best_vp = Viewport(x=vx, y=vy, w=viewport_w, h=viewport_h)

        if best_vp is None or best_count == 0:
            break

        # Remove covered cells
        newly_covered = set()
        for cy, cx in uncovered:
            if (best_vp.y <= cy < best_vp.y + best_vp.h and
                    best_vp.x <= cx < best_vp.x + best_vp.w):
                newly_covered.add((cy, cx))
        uncovered -= newly_covered
        placements.append(best_vp)

    return placements


def allocate_queries(
    num_seeds: int,
    viewport_placements: list[list[Viewport]],
    total_budget: int = 50,
    repeat_strategy: str = "depth",
) -> list[list[tuple[Viewport, int]]]:
    """Allocate query budget across seeds and viewports.

    Returns: per-seed list of (viewport, num_repeats) tuples.

    Strategies:
    - "coverage": spread queries across all viewports (old default)
    - "depth": prioritize repeats on the most important viewports
      (first few from greedy set-cover cover the most settlement-adjacent cells)
      This gives per-cell Monte Carlo estimates for Bayesian blending.
    """
    per_seed_budget = total_budget // num_seeds
    remainder = total_budget % num_seeds

    allocations = []
    for seed_idx in range(num_seeds):
        seed_budget = per_seed_budget + (1 if seed_idx < remainder else 0)
        vps = viewport_placements[seed_idx]

        if not vps:
            allocations.append([])
            continue

        if repeat_strategy == "coverage":
            # Old strategy: spread evenly
            per_vp = seed_budget // len(vps)
            vp_remainder = seed_budget % len(vps)
            seed_alloc = []
            for i, vp in enumerate(vps):
                repeats = per_vp + (1 if i < vp_remainder else 0)
                if repeats > 0:
                    seed_alloc.append((vp, repeats))
        else:
            # Depth strategy: first viewports (highest priority from greedy
            # set-cover) get more repeats. First covers most settlement-adjacent
            # cells. Target: 2-3 repeats on top viewports, 1 on the rest.
            n_vps = len(vps)
            # Top viewports get 2 repeats, rest get 1, until budget is spent
            seed_alloc = []
            budget_left = seed_budget
            for i, vp in enumerate(vps):
                if budget_left <= 0:
                    break
                # First 5 viewports get 2 repeats (they cover settlement clusters)
                if i < 5 and budget_left >= 2:
                    seed_alloc.append((vp, 2))
                    budget_left -= 2
                elif budget_left >= 1:
                    seed_alloc.append((vp, 1))
                    budget_left -= 1

        allocations.append(seed_alloc)

    return allocations


# ──────────────────────────────────────────────────────────────
# Probability normalization
# ──────────────────────────────────────────────────────────────

def normalize_prediction(tensor: np.ndarray, floor: float = PROB_FLOOR) -> np.ndarray:
    """Enforce minimum probability floor and renormalize to sum to 1.0.

    CRITICAL: Never allow 0.0 probabilities — KL divergence goes to infinity.

    We iterate clamp+renormalize to ensure the floor holds after division.
    In practice 2 passes suffice since clamping only increases mass.
    """
    for _ in range(3):
        tensor = np.maximum(tensor, floor)
        sums = tensor.sum(axis=-1, keepdims=True)
        tensor = tensor / sums
    # Final clamp to guarantee floor (renorm can only shrink by tiny epsilon)
    tensor = np.maximum(tensor, floor)
    sums = tensor.sum(axis=-1, keepdims=True)
    tensor = tensor / sums
    return tensor


# ──────────────────────────────────────────────────────────────
# Observation persistence (JSONL checkpoint)
# ──────────────────────────────────────────────────────────────

def get_obs_path(round_id: str) -> str:
    """Get the path to the observation checkpoint file."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, f"obs_{round_id}.jsonl")


def save_observation(round_id: str, observation: dict):
    """Append one observation to the JSONL checkpoint file."""
    path = get_obs_path(round_id)
    with open(path, "a") as f:
        json.dump(observation, f)
        f.write("\n")


def load_observations(round_id: str) -> list[dict]:
    """Load all observations from checkpoint file."""
    path = get_obs_path(round_id)
    if not os.path.exists(path):
        return []
    observations = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                observations.append(json.loads(line))
    return observations


# ──────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────

# Color scheme for terrain classes
CLASS_COLORS = [
    "#4a90d9",  # 0: Empty (blue-gray for ocean/plains)
    "#e6a332",  # 1: Settlement (amber)
    "#2ecc71",  # 2: Port (green)
    "#95a5a6",  # 3: Ruin (gray)
    "#27ae60",  # 4: Forest (dark green)
    "#8b4513",  # 5: Mountain (brown)
]


def plot_initial_grid(
    grid: list[list[int]],
    title: str = "Initial Grid",
    save_path: Optional[str] = None,
):
    """Render the initial terrain grid as a colored heatmap."""
    class_grid = grid_to_class_array(grid)
    cmap = ListedColormap(CLASS_COLORS)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(class_grid, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    cbar = plt.colorbar(im, ax=ax, ticks=range(NUM_CLASSES))
    cbar.ax.set_yticklabels(CLASS_NAMES)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.savefig(
            os.path.join(os.path.dirname(__file__), "data", f"{title.replace(' ', '_')}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


def plot_prediction_heatmap(
    prediction: np.ndarray,
    initial_grid: Optional[list[list[int]]] = None,
    title: str = "Prediction",
    save_path: Optional[str] = None,
):
    """Render prediction tensor as argmax grid + confidence heatmap side-by-side."""
    argmax = prediction.argmax(axis=-1)
    confidence = prediction.max(axis=-1)
    cmap = ListedColormap(CLASS_COLORS)

    ncols = 3 if initial_grid is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))

    col = 0
    if initial_grid is not None:
        class_grid = grid_to_class_array(initial_grid)
        im0 = axes[col].imshow(class_grid, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
        axes[col].set_title("Initial State")
        plt.colorbar(im0, ax=axes[col], ticks=range(NUM_CLASSES))
        col += 1

    im1 = axes[col].imshow(argmax, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
    axes[col].set_title(f"{title} — Argmax")
    plt.colorbar(im1, ax=axes[col], ticks=range(NUM_CLASSES))

    im2 = axes[col + 1].imshow(confidence, cmap="RdYlGn", vmin=0, vmax=1)
    axes[col + 1].set_title(f"{title} — Confidence")
    plt.colorbar(im2, ax=axes[col + 1])

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(__file__), "data",
            f"{title.replace(' ', '_')}.png",
        )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
