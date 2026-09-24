"""Tests for dyntmt.dynamic_marginal_msm."""

import math

from morie.fn import _array_core as np

from morie.fn.dyntmt import dynamic_marginal_msm


def test_dyntmt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    T = 5
    y = rng.normal(0, 1, n)
    D_history = rng.integers(0, 2, (n, T))
    H_history = rng.normal(0, 1, (n, T))
    regime_fn = lambda v: 1.0 if v > 0.0 else 0.0
    result = dynamic_marginal_msm(y, D_history, H_history, regime_fn)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert "n" in result
    assert result["n"] == n
    assert "n_time" in result
    assert result["n_time"] == T


def test_dyntmt_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 40
    T = 3
    y = rng.normal(0, 1, n)
    D_history = rng.integers(0, 2, (n, T))
    H_history = rng.normal(0, 1, (n, T))
    pres = rng.integers(0, 2, (n, T))
    result = dynamic_marginal_msm(y, D_history, H_history, pres)
    assert isinstance(result, dict)
    assert "n" in result
    assert result["n"] == n
    assert "n_time" in result
    assert result["n_time"] == T
    assert "value" in result
    assert math.isfinite(result["value"])
