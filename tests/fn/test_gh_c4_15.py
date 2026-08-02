"""Tests for gh_c4_15.ghosal_dp_mutual_sing."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_15 import ghosal_dp_mutual_sing


def test_gh_c4_15_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dp_mutual_sing(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c4_15_edge():
    """Test edge cases."""
    result = ghosal_dp_mutual_sing(np.array([42.0]))
    assert result["n"] == 1
