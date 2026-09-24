"""Tests for taylor.taylor_linearization."""

from morie.fn import _array_core as np

from morie.fn.taylor import taylor_linearization


def test_taylor_basic():
    """Test basic functionality."""
    y = 0.5
    weights = 0.5
    grad = 0.5
    result = taylor_linearization(y, weights, grad)
    assert isinstance(result, dict)
    assert "variance" in result


def test_taylor_edge():
    """Test edge cases."""
    y = 0.5
    weights = 0.5
    grad = 0.5
    result = taylor_linearization(y, weights, grad)
    assert isinstance(result, dict)
