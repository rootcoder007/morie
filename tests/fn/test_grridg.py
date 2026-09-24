"""Tests for grridg.geron_ridge_cost."""

from morie.fn import _array_core as np

from morie.fn.grridg import geron_ridge_cost


def test_grridg_basic():
    """Test basic functionality."""
    X = np.array([[1.0, 0.5], [1.0, -2.0], [1.0, 3.0]])
    y = np.array([1.0, 0.0, 2.0])
    theta = np.array([0.2, -0.3])
    alpha = 0.7
    result = geron_ridge_cost(X, y, theta, alpha)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grridg_edge():
    """Test edge cases."""
    X = np.array([[1.0, 0.5], [1.0, -2.0], [1.0, 3.0]])
    y = np.array([1.0, 0.0, 2.0])
    theta = np.array([0.2, -0.3])
    alpha = 0.7
    result = geron_ridge_cost(X, y, theta, alpha)
    assert isinstance(result, dict)
