"""Tests for kmnf4.kamath_nf4_datatype."""

from morie.fn import _array_core as np

from morie.fn.kmnf4 import kamath_nf4_datatype


def test_kmnf4_basic():
    """Test basic functionality."""
    result = kamath_nf4_datatype()
    assert isinstance(result, dict)
    assert "estimate" in result or "levels" in result


def test_kmnf4_edge():
    """Test edge cases."""
    result = kamath_nf4_datatype()
    assert isinstance(result, dict)
