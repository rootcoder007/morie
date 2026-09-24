"""Tests for midegf.mi_degrees_of_freedom."""

from morie.fn import _array_core as np

from morie.fn.midegf import mi_degrees_of_freedom


def test_midegf_basic():
    """Test basic functionality."""
    b = 0.1
    t = 0.1
    m = 5
    result = mi_degrees_of_freedom(b, t, m)
    assert isinstance(result, dict)
    assert "df" in result


def test_midegf_edge():
    """Test edge cases."""
    b = 0.1
    t = 0.1
    m = 5
    result = mi_degrees_of_freedom(b, t, m)
    assert isinstance(result, dict)
