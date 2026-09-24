"""Tests for regest.regression_estimator."""

import math

from morie.fn import _array_core as np

from morie.fn.regest import regression_estimator


def test_regest_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_x = np.random.default_rng(42)
    rng_w = np.random.default_rng(45)
    y = rng_y.normal(0, 1, 100)
    x = rng_x.normal(0, 1, 100)
    weights = rng_w.uniform(0.5, 2.0, 100)
    X_mean = 0.0  # known population mean of x
    result = regression_estimator(y, x, weights=weights, X_mean=X_mean)
    assert isinstance(result, dict)
    for key in ("mean", "slope", "intercept", "correlation",
                "variance_ratio_to_simple_mean", "passes_through_origin",
                "n", "method"):
        assert key in result
    assert math.isfinite(result["mean"])
    assert math.isfinite(result["slope"])
    assert math.isfinite(result["intercept"])
    assert -1.0 <= result["correlation"] <= 1.0
    assert 0.0 <= result["variance_ratio_to_simple_mean"] <= 1.0
    assert isinstance(result["passes_through_origin"], bool)
    assert result["n"] == 100
    assert isinstance(result["method"], str)


def test_regest_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    rng_x = np.random.default_rng(42)
    y = rng_y.normal(0, 1, 40)
    x = rng_x.normal(0, 1, 40)
    X_mean = 0.0
    # No weights supplied; default of None must be accepted.
    result = regression_estimator(y, x, X_mean=X_mean)
    assert isinstance(result, dict)
    for key in ("mean", "slope", "intercept", "correlation",
                "variance_ratio_to_simple_mean", "passes_through_origin",
                "n", "method"):
        assert key in result
    assert math.isfinite(result["mean"])
    assert result["n"] == 40
