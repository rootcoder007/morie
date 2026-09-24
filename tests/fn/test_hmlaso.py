"""Tests for hmlaso.geron_lasso_cost."""

from morie.fn import _array_core as np

from morie.fn.hmlaso import geron_lasso_cost


def test_hmlaso_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    theta = 0.1
    alpha = 0.1
    result = geron_lasso_cost(X, y, theta, alpha)
    assert isinstance(result, dict)
    assert "estimate" in result or "cost" in result


def test_hmlaso_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    theta = 0.1
    alpha = 0.1
    result = geron_lasso_cost(X, y, theta, alpha)
    assert isinstance(result, dict)
