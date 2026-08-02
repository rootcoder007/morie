"""Tests for gh_c8_14.ghosal_convex_misp."""

from morie.fn import _array_core as np

from morie.fn.gh_c8_14 import ghosal_convex_misp


def test_gh_c8_14_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_convex_misp(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c8_14_edge():
    """Test edge cases."""
    result = ghosal_convex_misp(np.array([42.0]))
    assert result["n"] == 1
