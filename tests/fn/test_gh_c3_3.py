"""Tests for gh_c3_3.ghosal_dir_simplex."""

from morie.fn import _array_core as np

from morie.fn.gh_c3_3 import ghosal_dir_simplex


def test_gh_c3_3_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dir_simplex(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c3_3_edge():
    """Test edge cases."""
    result = ghosal_dir_simplex(np.array([42.0]))
    assert result["n"] == 1
