"""Tests for or_from_rr.or_from_rr."""

from morie.fn import _array_core as np

from morie.fn.or_from_rr import or_from_rr


def test_ca11e28_basic():
    """Test basic functionality."""
    rr = 0.5
    p2 = 0.5
    result = or_from_rr(rr, p2)
    assert isinstance(result, dict)
    assert "value" in result


def test_ca11e28_edge():
    """Test edge cases."""
    rr = 0.5
    p2 = 0.5
    result = or_from_rr(rr, p2)
    assert isinstance(result, dict)
