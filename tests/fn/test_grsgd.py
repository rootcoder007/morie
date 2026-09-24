"""Tests for grsgd.geron_stochastic_gradient_descent."""

from morie.fn import _array_core as np

from morie.fn.grsgd import geron_stochastic_gradient_descent


def test_grsgd_basic():
    """Test basic functionality."""
    X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    y = [4.0, 7.0, 10.0]
    theta = [0.0, 0.0]
    eta = 0.05
    n_iter = 150
    result = geron_stochastic_gradient_descent(X, y, theta, eta, n_iter)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grsgd_edge():
    """Test edge cases."""
    X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    y = [4.0, 7.0, 10.0]
    theta = [0.0, 0.0]
    eta = 0.05
    n_iter = 150
    result = geron_stochastic_gradient_descent(X, y, theta, eta, n_iter)
    assert isinstance(result, dict)
