"""Tests for nbdsp.negative_binomial_dispersion."""

import math

from morie.fn import _array_core as np

from morie.fn.nbdsp import negative_binomial_dispersion


def test_nbdsp_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    p = 3
    y = rng.integers(0, 20, n)
    X = rng.normal(0, 1, (n, p))
    result = negative_binomial_dispersion(y, X, link="log")
    assert hasattr(result, "payload")
    assert math.isfinite(result.payload["estimate"])
    assert math.isfinite(result.payload["r_hat"])
    assert len(result.payload["mu_hat"]) == n
    assert len(result.payload["beta"]) == p
    assert result.payload["n"] == n


def test_nbdsp_edge():
    """Test edge cases with small but valid input."""
    rng = np.random.default_rng(7)
    n = 20
    p = 2
    y = rng.integers(0, 5, n)
    X = rng.normal(0, 1, (n, p))
    result = negative_binomial_dispersion(y, X, link="log", max_iter=50, tol=1e-8)
    assert "estimate" in result.payload
    assert "mu_hat" in result.payload
    assert "beta" in result.payload
    assert math.isfinite(result.payload["estimate"])
    assert len(result.payload["mu_hat"]) == n
    assert len(result.payload["beta"]) == p
