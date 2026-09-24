"""Tests for gpkrr.gp_kernel_ridge_reg."""

from morie.fn import _array_core as np

from morie.fn.gpkrr import gp_kernel_ridge_reg


def test_gpkrr_basic():
    """Test basic functionality."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    rng_t = np.random.default_rng(44)
    n, p, n_test = 40, 3, 10
    X = rng_x.normal(0, 1, (n, p))
    y = rng_y.normal(0, 1, n)
    X_test = rng_t.normal(0, 1, (n_test, p))
    result = gp_kernel_ridge_reg(X, y, X_test, lam=0.1)
    assert isinstance(result.payload, dict)
    assert "estimate" in result.payload
    assert "pred" in result.payload
    assert "var" in result.payload
    assert "alpha" in result.payload
    assert len(result.payload["pred"]) == n_test
    assert len(result.payload["var"]) == n_test
    assert len(result.payload["alpha"]) == n


def test_gpkrr_edge():
    """Test edge cases."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    rng_t = np.random.default_rng(44)
    n, p, n_test = 5, 2, 3
    X = rng_x.normal(0, 1, (n, p))
    y = rng_y.normal(0, 1, n)
    X_test = rng_t.normal(0, 1, (n_test, p))
    result = gp_kernel_ridge_reg(X, y, X_test, lam=0.1)
    assert isinstance(result.payload, dict)
    assert "pred" in result.payload
    assert "alpha" in result.payload
    assert len(result.payload["pred"]) == n_test
    assert len(result.payload["alpha"]) == n
