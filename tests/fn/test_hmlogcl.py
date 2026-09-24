"""Tests for hmlogcl.geron_logistic_cost."""

from morie.fn import _array_core as np

from morie.fn.hmlogcl import geron_logistic_cost


def test_hmlogcl_basic():
    """Test basic functionality."""
    X = 0.5
    y = 1
    theta = 0.5
    result = geron_logistic_cost(X, y, theta)
    assert isinstance(result, dict)
    assert "estimate" in result or "cost" in result


def test_hmlogcl_edge():
    """Test edge cases."""
    X = 0.5
    y = 1
    theta = 0.5
    result = geron_logistic_cost(X, y, theta)
    assert isinstance(result, dict)
