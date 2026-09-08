"""Tests for gh_c5_6.ghosal_vb_dpm."""

from morie.fn import _array_core as np

from morie.fn.gh_c5_6 import ghosal_vb_dpm


def test_gh_c5_6_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_vb_dpm(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c5_6_edge():
    """Test edge cases."""
    result = ghosal_vb_dpm(np.array([42.0]))
    assert result["n"] == 1
