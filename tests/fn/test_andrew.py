"""Tests for andrew.andrews_sine."""

from morie.fn import _array_core as np

from morie.fn.andrew import andrews_sine


def test_andrew_basic():
    """Test basic functionality."""
    r = 1.0
    result = andrews_sine(r)
    assert isinstance(result, dict)
    assert "weight" in result
def test_andrew_edge():
    """Test edge cases."""
    r = 1.0
    result = andrews_sine(r)
    assert isinstance(result, dict)
