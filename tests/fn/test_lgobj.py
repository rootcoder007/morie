"""Tests for lgobj.logistic_log_likelihood."""

import math

from morie.fn import _array_core as np

from morie.fn.lgobj import logistic_log_likelihood


def test_lgobj_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_X = np.random.default_rng(42)
    rng_b = np.random.default_rng(44)
    n, p = 40, 3
    y = list(rng_y.integers(0, 2, n))
    X = rng_X.normal(0, 1, (n, p))
    beta = list(rng_b.normal(0, 1, p))
    result = logistic_log_likelihood(y, X, beta)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "loglik" in result
    assert "p" in result
    assert "gradient" in result
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["loglik"])
    assert len(result["p"]) == n
    assert len(result["gradient"]) == p


def test_lgobj_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    rng_X = np.random.default_rng(42)
    rng_b = np.random.default_rng(44)
    n, p = 10, 2
    y = list(rng_y.integers(0, 2, n))
    X = rng_X.normal(0, 1, (n, p))
    beta = list(rng_b.normal(0, 1, p))
    result = logistic_log_likelihood(y, X, beta)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "loglik" in result
    assert "p" in result
    assert "gradient" in result
    assert len(result["p"]) == n
    assert len(result["gradient"]) == p
    assert math.isfinite(result["estimate"])
