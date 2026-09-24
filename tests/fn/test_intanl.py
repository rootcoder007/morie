"""Tests for intanl.interaction_analysis."""

import math

from morie.fn import _array_core as np

from morie.fn.intanl import interaction_analysis


def test_intanl_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    y = rng.normal(0, 1, 100)
    rng2 = np.random.default_rng(42)
    A = rng2.normal(0, 1, 100)
    V = rng2.normal(0, 1, 100)
    # Weights must be non-negative
    H = np.abs(rng2.normal(0, 1, 100))
    result = interaction_analysis(y, A, V, H)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "beta_av" in result
    assert "se" in result
    assert result["n"] == 100
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["se"])


def test_intanl_edge():
    """Test edge cases."""
    # Test with H=None (unweighted fit per docstring)
    rng = np.random.default_rng(43)
    y = rng.normal(0, 1, 100)
    rng2 = np.random.default_rng(42)
    A = rng2.normal(0, 1, 100)
    V = rng2.normal(0, 1, 100)
    result = interaction_analysis(y, A, V, None)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "beta_av" in result
    assert result["n"] == 100
    assert math.isfinite(result["estimate"])
