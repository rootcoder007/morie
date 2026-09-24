"""Tests for hrzdimr.horowitz_dimension_reduction."""

from morie.fn import _array_core as np

from morie.fn.hrzdimr import horowitz_dimension_reduction


def test_hrzdimr_basic():
    """Test basic functionality."""
    d = 5
    n = 5
    result = horowitz_dimension_reduction(d, n)
    assert isinstance(result, dict)
    assert "fullexp" in result


def test_hrzdimr_edge():
    """Test edge cases."""
    d = 5
    n = 5
    result = horowitz_dimension_reduction(d, n)
    assert isinstance(result, dict)
