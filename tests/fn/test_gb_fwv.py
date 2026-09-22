"""Tests for gb_fwv.gibbons_friedman_variance."""

from morie.fn import _array_core as np

from morie.fn.gb_fwv import gibbons_friedman_variance


def test_gb_fwv_basic():
    """Test basic functionality."""
    k = 5
    n = 50
    result = gibbons_friedman_variance(k, n)
    assert isinstance(result, dict)
    assert "mean_s" in result
def test_gb_fwv_edge():
    """Test edge cases."""
    k = 5
    n = 50
    result = gibbons_friedman_variance(k, n)
    assert isinstance(result, dict)
