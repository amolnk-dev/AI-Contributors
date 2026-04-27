"""Tests for utilities — grid parsing, viewport strategy, normalization."""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dtos import NUM_CLASSES, TERRAIN_TO_CLASS
from utils import (
    grid_to_class_array,
    identify_dynamic_cells,
    compute_viewport_placements,
    allocate_queries,
    normalize_prediction,
    save_observation,
    load_observations,
)


# ── Test: Grid parsing ───────────────────────────────────────

def test_grid_to_class_array_basic():
    """Grid codes should map correctly to class indices."""
    grid = [[10, 11, 0], [1, 2, 3], [4, 5, 11]]
    result = grid_to_class_array(grid)
    expected = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 0]])
    np.testing.assert_array_equal(result, expected)


def test_grid_to_class_array_unknown_code():
    """Unknown terrain codes should default to class 0."""
    grid = [[99]]
    result = grid_to_class_array(grid)
    assert result[0, 0] == 0


# ── Test: Dynamic cell identification ────────────────────────

def test_identify_dynamic_cells_static_only():
    """A grid with only ocean/plains/mountain should have no dynamic cells."""
    grid = [[10, 10, 10], [11, 11, 11], [5, 5, 5]]
    dynamic = identify_dynamic_cells(grid)
    assert len(dynamic) == 0


def test_identify_dynamic_cells_settlement():
    """Settlements and their neighborhoods should be dynamic."""
    grid = [[11, 11, 11, 11, 11],
            [11, 11, 11, 11, 11],
            [11, 11, 1, 11, 11],
            [11, 11, 11, 11, 11],
            [11, 11, 11, 11, 11]]
    settlements = [{"x": 2, "y": 2}]
    dynamic = identify_dynamic_cells(grid, settlements, radius=1)
    # Settlement itself + 8 neighbors (within radius 1)
    assert (2, 2) in dynamic
    assert (1, 1) in dynamic  # neighbor
    assert (3, 3) in dynamic  # neighbor


def test_identify_dynamic_cells_ocean_excluded():
    """Ocean cells should never be dynamic, even near settlements."""
    grid = [[10, 10, 10],
            [10, 1, 10],
            [10, 10, 10]]
    settlements = [{"x": 1, "y": 1}]
    dynamic = identify_dynamic_cells(grid, settlements, radius=1)
    # Only the settlement itself should be dynamic (ocean neighbors excluded)
    assert (1, 1) in dynamic
    assert (0, 0) not in dynamic  # ocean


# ── Test: Viewport placements ────────────────────────────────

def test_viewport_covers_all_dynamic():
    """Viewports should cover all dynamic cells."""
    # Create a grid with two clusters of dynamic cells
    dynamic = {(5, 5), (5, 6), (6, 5), (6, 6),   # cluster 1
               (30, 30), (30, 31), (31, 30)}       # cluster 2

    placements = compute_viewport_placements(
        dynamic, map_width=40, map_height=40, viewport_w=15, viewport_h=15,
    )

    # All dynamic cells should be covered
    covered = set()
    for vp in placements:
        for y in range(vp.y, vp.y + vp.h):
            for x in range(vp.x, vp.x + vp.w):
                covered.add((y, x))

    assert dynamic.issubset(covered)


def test_viewport_empty_dynamic():
    """No dynamic cells → no viewports needed."""
    placements = compute_viewport_placements(set())
    assert len(placements) == 0


# ── Test: Query allocation ───────────────────────────────────

def test_allocate_queries_even_distribution():
    """Budget should be distributed roughly evenly across seeds."""
    from dtos import Viewport
    vps = [[Viewport(x=0, y=0, w=15, h=15)]] * 5
    plan = allocate_queries(5, vps, total_budget=50)

    total = sum(r for seed_plan in plan for _, r in seed_plan)
    assert total == 50


def test_allocate_queries_no_viewports():
    """Seeds with no viewports should get 0 queries."""
    plan = allocate_queries(5, [[], [], [], [], []], total_budget=50)
    for seed_plan in plan:
        assert len(seed_plan) == 0


# ── Test: Observation persistence ────────────────────────────

def test_save_and_load_observations(tmp_path, monkeypatch):
    """Observations should round-trip through JSONL files."""
    monkeypatch.setattr(
        "utils.get_obs_path",
        lambda round_id: str(tmp_path / f"obs_{round_id}.jsonl"),
    )

    obs = {"seed_index": 0, "viewport": {"x": 0, "y": 0, "w": 15, "h": 15}, "grid": [[1, 2]]}
    save_observation("test-round", obs)
    save_observation("test-round", obs)

    loaded = load_observations("test-round")
    assert len(loaded) == 2
    assert loaded[0]["seed_index"] == 0


def test_load_observations_no_file():
    """Loading from non-existent file should return empty list."""
    result = load_observations("nonexistent-round-id-xyz")
    assert result == []
