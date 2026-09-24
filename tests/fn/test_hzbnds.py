"""Tests for hzbnds.horowitz_manski_bounds."""

from morie.fn import _array_core as np

from morie.fn.hzbnds import horowitz_manski_bounds


def test_hzbnds_basic():
    """Test basic functionality."""
    y = 0.5
    R = 1
    y_min = 0.5
    y_max = 0.5
    result = horowitz_manski_bounds(y, R, y_min, y_max)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_hzbnds_edge():
    """Test edge cases."""
    y = 0.5
    R = 1
    y_min = 0.5
    y_max = 0.5
    result = horowitz_manski_bounds(y, R, y_min, y_max)
    assert isinstance(result, dict)
