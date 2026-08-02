"""Tests for gh_c10_5.ghosal_wn_adapt."""

from morie.fn import _array_core as np

from morie.fn.gh_c10_5 import ghosal_wn_adapt


def test_gh_c10_5_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_wn_adapt(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c10_5_edge():
    """Test edge cases."""
    result = ghosal_wn_adapt(np.array([42.0]))
    assert result["n"] == 1
