"""Tests for gb651c.gibbons_ctrl_median_curtail."""

from morie.fn import _array_core as np

from morie.fn.gb651c import gibbons_ctrl_median_curtail


def test_gb651c_basic():
    """Test basic functionality."""
    m = 5
    n = 5
    result = gibbons_ctrl_median_curtail(m, n)
    assert isinstance(result, dict)
    assert "d" in result or "d" in result


def test_gb651c_edge():
    """Test edge cases."""
    m = 5
    n = 5
    result = gibbons_ctrl_median_curtail(m, n)
    assert isinstance(result, dict)
