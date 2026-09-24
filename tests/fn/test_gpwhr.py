"""Tests for gpwhr.gp_warped."""

import math

from morie.fn import _array_core as np

from morie.fn.gpwhr import gp_warped


def test_gpwhr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    X_test = rng.normal(0, 1, (10, 3))
    result = gp_warped(X, y, X_test, warp="identity")
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "median" in result
    assert len(result["median"]) == 10
    assert math.isfinite(result["estimate"])


def test_gpwhr_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    X_test = rng.normal(0, 1, (5, 3))
    result = gp_warped(X, y, X_test, warp="identity")
    assert isinstance(result, dict)
    assert "estimate" in result
    assert len(result["median"]) == 5
    assert math.isfinite(result["estimate"])
