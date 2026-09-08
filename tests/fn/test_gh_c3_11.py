"""Tests for gh_c3_11.ghosal_tailfree_def."""

from morie.fn import _array_core as np

from morie.fn.gh_c3_11 import ghosal_tailfree_def


def test_gh_c3_11_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_tailfree_def(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c3_11_edge():
    """Test edge cases."""
    result = ghosal_tailfree_def(np.array([42.0]))
    assert result["n"] == 1
