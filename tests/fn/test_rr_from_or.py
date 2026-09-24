"""Tests for rr_from_or.rr_from_or."""

from morie.fn import _array_core as np

from morie.fn.rr_from_or import rr_from_or


def test_ca11e29_basic():
    """Test basic functionality."""
    or_value = 0.5
    p2 = 0.5
    result = rr_from_or(or_value, p2)
    assert isinstance(result, dict)
    assert "value" in result


def test_ca11e29_edge():
    """Test edge cases."""
    or_value = 0.5
    p2 = 0.5
    result = rr_from_or(or_value, p2)
    assert isinstance(result, dict)
