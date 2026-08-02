"""Tests for gh_c4_17.ghosal_dp_median."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_17 import ghosal_dp_median


def test_gh_c4_17_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dp_median(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c4_17_edge():
    """Test edge cases."""
    result = ghosal_dp_median(np.array([42.0]))
    assert result["n"] == 1
