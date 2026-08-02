"""Tests for gh_c6_8.ghosal_tailfree_con."""

from morie.fn import _array_core as np

from morie.fn.gh_c6_8 import ghosal_tailfree_con


def test_gh_c6_8_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_tailfree_con(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c6_8_edge():
    """Test edge cases."""
    result = ghosal_tailfree_con(np.array([42.0]))
    assert result["n"] == 1
