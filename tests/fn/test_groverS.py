"""Tests for groverS.grover_search."""

import math

from morie.fn import _array_core as np

from morie.fn.groverS import grover_search


def test_groverS_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    oracle = list(rng.integers(0, 2, n))
    # Ensure at least one marked and one unmarked
    if sum(oracle) == 0:
        oracle[0] = 1
    if sum(oracle) == n:
        oracle[0] = 0
    result = grover_search(oracle, n)
    assert isinstance(result, dict)
    assert "p_success" in result
    assert "k_opt" in result
    assert "theta" in result
    assert 0.0 <= result["p_success"] <= 1.0
    assert math.isfinite(result["theta"])
    assert isinstance(result["k_opt"], int)
    assert result["k_opt"] >= 1


def test_groverS_edge():
    """Test edge cases."""
    n = 4
    oracle = [1, 0, 0, 0]
    result = grover_search(oracle, n)
    assert isinstance(result, dict)
    assert "p_success" in result
    assert "k_opt" in result
    assert "p_closed_form" in result
    assert 0.0 <= result["p_success"] <= 1.0
    assert isinstance(result["k_opt"], int)
    assert result["k_opt"] >= 1
