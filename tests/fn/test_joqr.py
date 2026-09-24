"""Tests for joqr.joseph_quantile_regression."""

import math

from morie.fn import _array_core as np

from morie.fn.joqr import joseph_quantile_regression


def test_joqr_basic():
    """Test basic functionality."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    X = rng_x.normal(0, 1, (100, 5))
    y = rng_y.normal(0, 1, 100)
    tau = 0.1
    result = joseph_quantile_regression(X, y, tau)
    # result exposes attributes: intercept, loss, q, n
    assert hasattr(result, "intercept")
    assert hasattr(result, "loss")
    assert hasattr(result, "q")
    assert hasattr(result, "n")
    assert math.isfinite(float(result.intercept))
    assert math.isfinite(float(result.loss))


def test_joqr_edge():
    """Test edge cases."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    X = rng_x.normal(0, 1, (100, 5))
    y = rng_y.normal(0, 1, 100)
    tau = 0.1
    result = joseph_quantile_regression(X, y, tau)
    assert hasattr(result, "q")
    assert hasattr(result, "n")
    assert math.isfinite(float(result.q))
    assert int(result.n) == len(y)
