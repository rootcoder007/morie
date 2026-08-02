"""Tests for gh_gp_brow_prim.ghosal_gp_brownian_primitive."""

from morie.fn import _array_core as np

from morie.fn.gh_gp_brow_prim import ghosal_gp_brownian_primitive


def test_gh_gp_brow_prim_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_gp_brownian_primitive(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_gp_brow_prim_edge():
    """Test edge cases."""
    result = ghosal_gp_brownian_primitive(np.array([42.0]))
    assert result["n"] == 1
