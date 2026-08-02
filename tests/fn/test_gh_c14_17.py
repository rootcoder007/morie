"""Tests for gh_c14_17.ghosal_disc_rp_rel."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_17 import ghosal_disc_rp_rel


def test_gh_c14_17_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_disc_rp_rel(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c14_17_edge():
    """Test edge cases."""
    result = ghosal_disc_rp_rel(np.array([42.0]))
    assert result["n"] == 1
