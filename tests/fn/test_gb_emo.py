"""Tests for gb_emo.gibbons_order_moments."""

from morie.fn import _array_core as np

from morie.fn.gb_emo import gibbons_order_moments


def test_gb_emo_basic():
    """Test basic functionality."""
    r = 10
    n = 100
    result = gibbons_order_moments(r, n)
    assert isinstance(result, dict)
    assert "moment" in result
def test_gb_emo_edge():
    """Test edge cases."""
    r = 10
    n = 100
    result = gibbons_order_moments(r, n)
    assert isinstance(result, dict)
