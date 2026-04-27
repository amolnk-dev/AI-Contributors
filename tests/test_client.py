"""Tests for the HTTP client — budget tracking and auth detection."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from client import AstarClient, BudgetExhaustedError, AuthExpiredError


def test_client_requires_token():
    """Client should raise ValueError if no token provided."""
    # Clear env var if set
    old = os.environ.pop("ASTAR_TOKEN", None)
    try:
        with pytest.raises(ValueError, match="No auth token"):
            AstarClient(token="")
    finally:
        if old:
            os.environ["ASTAR_TOKEN"] = old


def test_budget_exhaustion_raises():
    """Client should raise BudgetExhaustedError when budget is exhausted."""
    client = AstarClient(token="test-token")
    # Simulate exhausted budget
    client._queries_used = 50
    client._queries_max = 50

    with pytest.raises(BudgetExhaustedError, match="Budget exhausted"):
        client.simulate(
            round_id="fake-id",
            seed_index=0,
            viewport_x=0,
            viewport_y=0,
        )


def test_budget_tracking_initial():
    """Budget should start at 0 used, 50 max."""
    client = AstarClient(token="test-token")
    assert client.queries_used == 0
    assert client.queries_remaining == 50


def test_queries_remaining_property():
    """queries_remaining should reflect current state."""
    client = AstarClient(token="test-token")
    client._queries_used = 23
    client._queries_max = 50
    assert client.queries_remaining == 27
