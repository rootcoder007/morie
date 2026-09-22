"""Tests for gb_rz.gibbons_rz_test."""

from morie.fn import _array_core as np

from morie.fn.gb_rz import gibbons_rz_test


def test_gb_rz_basic():
    """Test basic functionality."""
    pmf = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_rz_test(pmf)
    assert isinstance(result, dict)
    assert "gamma" in result
def test_gb_rz_edge():
    """Test edge cases."""
    pmf = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_rz_test(pmf)
    assert isinstance(result, dict)
