"""Tests for manski.manski_no_assumption_bounds."""

from morie.fn import _array_core as np

from morie.fn.manski import manski_no_assumption_bounds


def test_manski_basic():
    """Test basic functionality."""
    y = 0.5
    D = 1
    y_min = 0.5
    y_max = 5
    result = manski_no_assumption_bounds(y, D, y_min, y_max)
    assert isinstance(result, dict)
    assert "ate_lower" in result


def test_manski_edge():
    """Test edge cases."""
    y = 0.5
    D = 1
    y_min = 0.5
    y_max = 5
    result = manski_no_assumption_bounds(y, D, y_min, y_max)
    assert isinstance(result, dict)
