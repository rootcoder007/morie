"""Tests for gb_t1e.gibbons_type1_error."""

from morie.fn import _array_core as np

from morie.fn.gb_t1e import gibbons_type1_error


def test_gb_t1e_basic():
    """Test basic functionality."""
    pmf = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_type1_error(pmf)
    assert isinstance(result, dict)
    assert "sizes" in result
def test_gb_t1e_edge():
    """Test edge cases."""
    pmf = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_type1_error(pmf)
    assert isinstance(result, dict)
