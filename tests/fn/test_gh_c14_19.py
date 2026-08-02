"""Tests for gh_c14_19.ghosal_local_dp."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_19 import ghosal_local_dp


def test_gh_c14_19_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_local_dp(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c14_19_edge():
    """Test edge cases."""
    result = ghosal_local_dp(np.array([42.0]))
    assert result["n"] == 1
