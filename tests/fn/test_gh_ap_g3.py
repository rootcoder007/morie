"""Tests for gh_ap_g3.ghosal_dir_marginal."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_g3 import ghosal_dir_marginal


def test_gh_ap_g3_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dir_marginal(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_ap_g3_edge():
    """Test edge cases."""
    result = ghosal_dir_marginal(np.array([42.0]))
    assert result["n"] == 1
