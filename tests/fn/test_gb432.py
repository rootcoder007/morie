"""Tests for gb432.gibbons_ks_exact_dist."""

from morie.fn import _array_core as np

from morie.fn.gb432 import gibbons_ks_exact_dist


def test_gb432_basic():
    """Test basic functionality."""
    d = 3
    n = 100
    result = gibbons_ks_exact_dist(d, n)
    assert isinstance(result, dict)
    assert "cdf" in result
def test_gb432_edge():
    """Test edge cases."""
    d = 3
    n = 100
    result = gibbons_ks_exact_dist(d, n)
    assert isinstance(result, dict)
