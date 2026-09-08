"""Tests for gh_c7_6.ghosal_pt_dens_con."""

from morie.fn import _array_core as np

from morie.fn.gh_c7_6 import ghosal_pt_dens_con


def test_gh_c7_6_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_pt_dens_con(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c7_6_edge():
    """Test edge cases."""
    result = ghosal_pt_dens_con(np.array([42.0]))
    assert result["n"] == 1
