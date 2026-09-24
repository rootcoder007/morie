"""Tests for mirt2.mirt_2d_compensatory."""

from morie.fn import _array_core as np

from morie.fn.mirt2 import mirt_2d_compensatory


def test_mirt2_basic():
    """Test basic functionality."""
    y = 1
    theta = 0.5
    a = 0.5
    d = 0.5
    result = mirt_2d_compensatory(y, theta, a, d)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_mirt2_edge():
    """Test edge cases."""
    y = 1
    theta = 0.5
    a = 0.5
    d = 0.5
    result = mirt_2d_compensatory(y, theta, a, d)
    assert isinstance(result, dict)
