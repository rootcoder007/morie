"""Tests for dplog.dp_logistic."""

import math

import pytest
from morie.fn import _array_core as np

from morie.fn.dplog import dp_logistic


def test_dplog_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    X = rng.normal(0, 1, (n, p))
    y = rng.integers(0, 2, n)
    result = dp_logistic(X, y, epsilon=1.0, method="objective", seed=0)
    assert isinstance(result, dict)
    assert "beta" in result
    assert "accuracy" in result
    assert "clipped_fraction" in result
    assert "epsilon" in result
    assert len(result["beta"]) == p
    assert math.isfinite(result["accuracy"])
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["clipped_fraction"] <= 1.0
    assert result["epsilon"] == 1.0


def test_dplog_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    X = rng.normal(0, 1, (n, p))
    y = rng.integers(0, 2, n)
    # Objective perturbation requires lam > 0 for strong convexity
    with pytest.raises(ValueError):
        dp_logistic(X, y, epsilon=1.0, method="objective", lam=0.0, seed=1)
