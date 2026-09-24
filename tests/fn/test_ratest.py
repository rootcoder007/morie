"""Tests for ratest.ratio_estimator."""

import math

from morie.fn import _array_core as np

from morie.fn.ratest import ratio_estimator


def test_ratest_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_x = np.random.default_rng(42)
    rng_w = np.random.default_rng(45)
    n = 100
    y = rng_y.normal(0, 1, n)
    x = rng_x.normal(0, 1, n)
    weights = np.array(rng_w.uniform(0.5, 1.5, n))
    X_total = float(np.sum(x))
    result = ratio_estimator(y, x, weights, X_total=X_total)
    assert isinstance(result, dict)
    assert "ratio" in result
    assert "mean" in result
    assert "total" in result
    assert "n" in result
    assert result["n"] == n
    assert math.isfinite(result["ratio"])


def test_ratest_edge():
    """Test edge case with minimal sample size."""
    rng = np.random.default_rng(7)
    n = 3
    y = rng.normal(0, 1, n)
    x = rng.normal(0, 1, n)
    X_total = float(np.sum(x))
    result = ratio_estimator(y, x, X_total=X_total)
    assert isinstance(result, dict)
    assert result["n"] == n
    assert math.isfinite(result["ratio"])
