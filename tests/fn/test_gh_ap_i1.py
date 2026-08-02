"""Tests for gh_ap_i1.ghosal_gp_sample_cont."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_i1 import ghosal_gp_sample_cont


def test_gh_ap_i1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_gp_sample_cont(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_ap_i1_edge():
    """Test edge cases."""
    result = ghosal_gp_sample_cont(np.array([42.0]))
    assert result["n"] == 1
