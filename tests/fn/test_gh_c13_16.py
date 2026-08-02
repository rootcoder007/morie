"""Tests for gh_c13_16.ghosal_bb_censored."""

from morie.fn import _array_core as np

from morie.fn.gh_c13_16 import ghosal_bb_censored


def test_gh_c13_16_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_bb_censored(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c13_16_edge():
    """Test edge cases."""
    result = ghosal_bb_censored(np.array([42.0]))
    assert result["n"] == 1
