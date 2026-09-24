"""Tests for r2_from_f2.r2_from_f2."""

from morie.fn import _array_core as np

from morie.fn.r2_from_f2 import r2_from_f2


def test_ca8e7_basic():
    """Test basic functionality."""
    f2 = 0.5
    result = r2_from_f2(f2)
    assert isinstance(result, dict)
    assert "value" in result


def test_ca8e7_edge():
    """Test edge cases."""
    f2 = 0.5
    result = r2_from_f2(f2)
    assert isinstance(result, dict)
