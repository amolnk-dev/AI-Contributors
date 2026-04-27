"""Tests for the prediction model — critical path tests.

Priority 1 (blocks submission):
- test_prediction_shape: wrong shape → 400 error on submit
- test_normalize_no_zeros: zeros → infinite KL divergence
- test_normalize_sums_to_one: bad sums → 400 error on submit
- test_terrain_to_class_all_codes: wrong mapping → wrong predictions
- test_static_prediction_deterministic: static cells must be correct
"""

import sys
import os

import numpy as np
import pytest

# Add task directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dtos import TERRAIN_TO_CLASS, NUM_CLASSES, STATIC_TERRAIN_CODES, PROB_FLOOR
from model import (
    build_static_prediction,
    update_with_observations,
    build_prediction,
)
from utils import normalize_prediction, grid_to_class_array


# ── Sample data ──────────────────────────────────────────────

def _make_simple_grid(h=10, w=10):
    """Create a simple test grid with known terrain."""
    grid = [[11] * w for _ in range(h)]  # All plains

    # Add borders of ocean
    for y in range(h):
        grid[y][0] = 10
        grid[y][w - 1] = 10
    for x in range(w):
        grid[0][x] = 10
        grid[h - 1][x] = 10

    # Add some features
    grid[3][3] = 5   # Mountain
    grid[4][4] = 1   # Settlement
    grid[4][5] = 4   # Forest
    grid[5][4] = 2   # Port (near ocean? not really, but for testing)
    grid[6][6] = 3   # Ruin

    return grid


def _make_settlements():
    return [
        {"x": 4, "y": 4, "has_port": False, "alive": True},
        {"x": 4, "y": 5, "has_port": True, "alive": True},
    ]


# ── Test: Terrain mapping ────────────────────────────────────

def test_terrain_to_class_all_codes():
    """All terrain codes must map to valid class indices [0, 5]."""
    for code, cls in TERRAIN_TO_CLASS.items():
        assert 0 <= cls < NUM_CLASSES, f"Code {code} maps to invalid class {cls}"


def test_terrain_to_class_ocean_plains_empty_collapse():
    """Ocean (10), Plains (11), and Empty (0) all map to class 0."""
    assert TERRAIN_TO_CLASS[0] == 0
    assert TERRAIN_TO_CLASS[10] == 0
    assert TERRAIN_TO_CLASS[11] == 0


def test_terrain_to_class_specific_mappings():
    """Verify specific terrain → class mappings."""
    assert TERRAIN_TO_CLASS[1] == 1  # Settlement
    assert TERRAIN_TO_CLASS[2] == 2  # Port
    assert TERRAIN_TO_CLASS[3] == 3  # Ruin
    assert TERRAIN_TO_CLASS[4] == 4  # Forest
    assert TERRAIN_TO_CLASS[5] == 5  # Mountain


# ── Test: Prediction shape ───────────────────────────────────

def test_prediction_shape():
    """Prediction tensor must be exactly H×W×6."""
    grid = _make_simple_grid(40, 40)
    tensor = build_static_prediction(grid)
    assert tensor.shape == (40, 40, NUM_CLASSES)


def test_prediction_shape_small():
    """Works with non-standard grid sizes."""
    grid = _make_simple_grid(10, 10)
    tensor = build_static_prediction(grid)
    assert tensor.shape == (10, 10, NUM_CLASSES)


# ── Test: Normalization ──────────────────────────────────────

def test_normalize_no_zeros():
    """After normalization, no probability should be 0.0.

    CRITICAL: KL divergence → infinity if ground truth has p>0 and we have q=0.
    """
    # Start with a tensor that has zeros
    tensor = np.zeros((5, 5, NUM_CLASSES))
    tensor[:, :, 0] = 1.0  # All probability on class 0

    normalized = normalize_prediction(tensor)

    # No zeros anywhere (allow tiny floating-point epsilon below floor)
    assert normalized.min() >= PROB_FLOOR * 0.99, (
        f"Found probability below floor: {normalized.min()}"
    )


def test_normalize_sums_to_one():
    """Each cell's 6 probabilities must sum to 1.0 (within tolerance)."""
    tensor = np.random.rand(40, 40, NUM_CLASSES)
    normalized = normalize_prediction(tensor)

    sums = normalized.sum(axis=-1)
    np.testing.assert_allclose(sums, 1.0, atol=0.01)


def test_normalize_preserves_relative_order():
    """Normalization should preserve the relative ranking of probabilities."""
    tensor = np.array([[[0.5, 0.3, 0.1, 0.05, 0.03, 0.02]]])
    normalized = normalize_prediction(tensor)

    # Original order should be preserved
    assert normalized[0, 0, 0] > normalized[0, 0, 1] > normalized[0, 0, 2]


def test_normalize_handles_all_zeros():
    """Edge case: all-zero tensor should get uniform distribution after floor."""
    tensor = np.zeros((3, 3, NUM_CLASSES))
    normalized = normalize_prediction(tensor)

    # Should be approximately uniform after floor + renormalize
    expected = PROB_FLOOR / (PROB_FLOOR * NUM_CLASSES)
    np.testing.assert_allclose(normalized, expected, atol=0.01)


# ── Test: Static prediction ─────────────────────────────────

def test_static_prediction_ocean():
    """Ocean cells should predict class 0 with ~1.0 probability."""
    grid = [[10, 10], [10, 10]]
    tensor = build_static_prediction(grid)
    # Ocean → class 0 = 1.0, all others = 0.0
    assert tensor[0, 0, 0] == 1.0
    assert tensor[0, 0, 1] == 0.0


def test_static_prediction_mountain():
    """Mountain cells should predict class 5 with ~1.0 probability."""
    grid = [[5, 5], [5, 5]]
    tensor = build_static_prediction(grid)
    # Mountain → class 5 = 1.0
    assert tensor[0, 0, 5] == 1.0
    assert tensor[0, 0, 0] == 0.0


def test_static_prediction_settlement_not_deterministic():
    """Settlement cells should NOT be deterministic — they can change."""
    grid = [[1]]
    tensor = build_static_prediction(grid)
    # Settlement should have probability spread across multiple classes
    assert tensor[0, 0, 1] > 0  # Some P(settlement)
    assert tensor[0, 0, 3] > 0  # Some P(ruin)
    # Should not be 1.0 for any single class
    assert tensor[0, 0].max() < 1.0


# ── Test: Full pipeline (mocked) ────────────────────────────

def test_full_pipeline_produces_valid_tensor():
    """End-to-end: build_prediction should produce a valid submission tensor."""
    grid = _make_simple_grid(10, 10)
    settlements = _make_settlements()

    # Fake observation
    observations = [{
        "seed_index": 0,
        "viewport": {"x": 2, "y": 2, "w": 6, "h": 6},
        "grid": [[11, 11, 1, 4, 11, 11],
                 [11, 4, 11, 11, 11, 11],
                 [11, 11, 11, 11, 11, 11],
                 [11, 11, 2, 11, 11, 11],
                 [11, 11, 11, 3, 11, 11],
                 [11, 11, 11, 11, 11, 11]],
        "settlements": [],
    }]

    prediction = build_prediction(
        initial_grid=grid,
        settlements=settlements,
        observations=observations,
        seed_index=0,
        all_initial_grids=[grid],
        all_observations=observations,
        calibration=None,
    )

    # Shape check
    assert prediction.shape == (10, 10, NUM_CLASSES)

    # No zeros (allow tiny floating-point epsilon below floor)
    assert prediction.min() >= PROB_FLOOR * 0.99

    # Sums to 1.0
    sums = prediction.sum(axis=-1)
    np.testing.assert_allclose(sums, 1.0, atol=0.01)

    # All probabilities non-negative
    assert (prediction >= 0).all()
