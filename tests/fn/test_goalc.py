"""Tests for goalc.goal_conditioned."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.goalc import goal_conditioned


def test_goalc_basic():
    """Test basic functionality."""
    # Deterministic transition list: chain 0 -> 1 -> 2 -> 3
    env = [[0, 0, 1], [1, 1, 2], [2, 0, 3]]
    result = goal_conditioned(env, n_states=4)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "v" in result
    assert "expected_value" in result
    assert "reachable" in result
    # V should have one row per state
    assert len(result["v"]) == 4
    # reachable is a share in [0, 1]
    assert 0.0 <= result["reachable"] <= 1.0


def test_goalc_edge():
    """Test edge cases."""
    # Two-state environment with bidirectional transitions
    env = [[0, 0, 1], [1, 0, 0]]
    result = goal_conditioned(env, n_states=2, gamma=0.9, step_cost=-1.0)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["reachable"] <= 1.0
    assert len(result["v"]) == 2
