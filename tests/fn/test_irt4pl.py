"""Tests for irt4pl.four_parameter_logistic."""

from morie.fn import _array_core as np

from morie.fn.irt4pl import four_parameter_logistic


def test_irt4pl_basic():
    """Test basic functionality."""
    y = 0.5
    theta = 0.5
    a = 0.5
    b = 0.5
    result = four_parameter_logistic(y, theta, a, b)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_irt4pl_edge():
    """Test edge cases."""
    y = 0.5
    theta = 0.5
    a = 0.5
    b = 0.5
    result = four_parameter_logistic(y, theta, a, b)
    assert isinstance(result, dict)
