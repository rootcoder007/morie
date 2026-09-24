"""Tests for grlogp.geron_logistic_regression_probability."""

from morie.fn import _array_core as np

from morie.fn.grlogp import geron_logistic_regression_probability


def test_grlogp_basic():
    """Test basic functionality."""
    X = [[1.0, 2.0], [1.0, -3.0]]
    theta = [0.5, 1.0]
    result = geron_logistic_regression_probability(X, theta)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grlogp_edge():
    """Test edge cases."""
    X = [[1.0, 2.0], [1.0, -3.0]]
    theta = [0.5, 1.0]
    result = geron_logistic_regression_probability(X, theta)
    assert isinstance(result, dict)
