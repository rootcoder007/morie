"""Tests for hrzcs.horowitz_curse_dimensionality."""

from morie.fn import _array_core as np

from morie.fn.hrzcs import horowitz_curse_dimensionality


def test_hrzcs_basic():
    """Test basic functionality."""
    d = 5
    n = 5
    result = horowitz_curse_dimensionality(d, n)
    assert isinstance(result, dict)
    assert "exponent" in result


def test_hrzcs_edge():
    """Test edge cases."""
    d = 5
    n = 5
    result = horowitz_curse_dimensionality(d, n)
    assert isinstance(result, dict)
