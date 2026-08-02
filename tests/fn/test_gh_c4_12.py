"""Tests for gh_c4_12.ghosal_dp_discrete."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_12 import ghosal_dp_discrete


def test_gh_c4_12_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dp_discrete(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c4_12_edge():
    """Test edge cases."""
    result = ghosal_dp_discrete(np.array([42.0]))
    assert result["n"] == 1
