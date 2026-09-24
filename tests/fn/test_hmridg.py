"""Tests for hmridg.geron_ridge_cost."""

from morie.fn import _array_core as np

from morie.fn.hmridg import geron_ridge_cost


def test_hmridg_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    theta = 0.1
    alpha = 0.1
    result = geron_ridge_cost(X, y, theta, alpha)
    assert isinstance(result, dict)
    assert "estimate" in result or "cost" in result


def test_hmridg_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    theta = 0.1
    alpha = 0.1
    result = geron_ridge_cost(X, y, theta, alpha)
    assert isinstance(result, dict)
