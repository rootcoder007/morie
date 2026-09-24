"""Tests for manct.ma_continuity_correction."""

from morie.fn import _array_core as np

from morie.fn.manct import ma_continuity_correction


def test_manct_basic():
    """Test basic functionality."""
    a = 0.5
    b = 0.5
    c = 0.5
    d = 0.5
    result = ma_continuity_correction(a, b, c, d)
    assert isinstance(result, dict)
    assert "estimate" in result or "a_adj" in result


def test_manct_edge():
    """Test edge cases."""
    a = 0.5
    b = 0.5
    c = 0.5
    d = 0.5
    result = ma_continuity_correction(a, b, c, d)
    assert isinstance(result, dict)
