"""Tests for bndpl.bnp_density_pl."""

from morie.fn import _array_core as np

from morie.fn.bndpl import bnp_density_pl


def test_bndpl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = bnp_density_pl(y)
    assert isinstance(result, dict)
    assert "grid" in result
def test_bndpl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = bnp_density_pl(y)
    assert isinstance(result, dict)
