"""Tests for gpmoe.gp_mixture_of_experts."""

from morie.fn import _array_core as np

from morie.fn.gpmoe import gp_mixture_of_experts


def test_gpmoe_basic():
    """Test basic functionality."""
    rng_X = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    rng_xt = np.random.default_rng(44)

    n, p, m, K = 40, 3, 10, 3
    X = rng_X.normal(0, 1, (n, p))
    y = rng_y.normal(0, 1, n)
    X_test = rng_xt.normal(0, 1, (m, p))

    result = gp_mixture_of_experts(X, y, X_test, K)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "mean" in result
    assert "var" in result
    assert "gate" in result
    assert len(result["mean"]) == m
    assert len(result["var"]) == m
    assert len(result["gate"]) == m
    assert len(result["gate"][0]) == K


def test_gpmoe_edge():
    """Test edge cases."""
    rng_X = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    rng_xt = np.random.default_rng(44)

    n, p, m, K = 40, 3, 10, 2
    X = rng_X.normal(0, 1, (n, p))
    y = rng_y.normal(0, 1, n)
    X_test = rng_xt.normal(0, 1, (m, p))

    result = gp_mixture_of_experts(X, y, X_test, K, ell=2.0, noise=1e-5)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "mean" in result
    assert "var" in result
    assert "gate" in result
    assert len(result["mean"]) == m
    assert len(result["var"]) == m
    assert len(result["gate"]) == m
    assert len(result["gate"][0]) == K
