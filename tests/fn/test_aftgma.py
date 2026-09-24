"""Tests for aftgma.aft_generalized_gamma."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.aftgma import aft_generalized_gamma


def test_aftgma_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    p = 3
    X = rng.normal(0, 1, (n, p))
    X0 = X[:, 0]
    X1 = X[:, 1]
    X2 = X[:, 2]
    mu = [1.0 + 0.5 * X0[i] - 0.3 * X1[i] + 0.2 * X2[i] for i in range(n)]
    eps = rng.normal(0, 1, n)
    T = [math.exp(mu[i] + 0.5 * eps[i]) for i in range(n)]
    C = rng.uniform(0.5, 10.0, n)
    time = [min(T[i], C[i]) for i in range(n)]
    event = [1.0 if T[i] <= C[i] else 0.0 for i in range(n)]
    result = aft_generalized_gamma(time, event, X)
    assert isinstance(result, dict)
    for key in ("beta", "time_ratio", "sigma", "loglik", "aic", "converged"):
        assert key in result
    assert len(result["beta"]) == p + 1
    assert len(result["time_ratio"]) == p + 1
    assert math.isfinite(float(result["sigma"]))
    assert math.isfinite(float(result["loglik"]))
    assert float(result["sigma"]) > 0


def test_aftgma_edge():
    """Test that zero-time inputs raise ValueError as documented."""
    with pytest.raises(ValueError):
        aft_generalized_gamma([0.0, 1.0], [1.0, 1.0], [[1.0], [2.0]])
