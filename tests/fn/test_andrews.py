"""Tests for andrews.andrews_sine."""

from morie.fn import _array_core as np

from morie.fn.andrews import andrews_sine


def test_andrews_basic():
    """Test basic functionality."""
    r = 10
    result = andrews_sine(r)
    assert isinstance(result, dict)
    assert "psi" in result
def test_andrews_edge():
    """Test edge cases."""
    r = 10
    result = andrews_sine(r)
    assert isinstance(result, dict)
