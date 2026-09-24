"""Tests for hillEst.hill_estimator."""

import math

from morie.fn import _array_core as np

from morie.fn.hillEst import hill_estimator


def test_hillEst_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    k = 5
    result = hill_estimator(x, k)
    assert isinstance(result, dict)
    assert "xi" in result
    xi = result["xi"]
    assert math.isfinite(xi)
    assert xi > 0  # according to docstring, xi > 0


def test_hillEst_edge():
    """Test edge cases with default k."""
    rng = np.random.default_rng(123)
    x = rng.normal(0, 1, 50)
    result = hill_estimator(x)  # k defaults to None
    assert isinstance(result, dict)
    assert "xi" in result
    xi = result["xi"]
    assert math.isfinite(xi)
    assert xi > 0
