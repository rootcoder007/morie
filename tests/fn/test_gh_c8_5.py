"""Tests for gh_c8_5.ghosal_entropy_cnd."""

from morie.fn import _array_core as np

from morie.fn.gh_c8_5 import ghosal_entropy_cnd


def test_gh_c8_5_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_entropy_cnd(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c8_5_edge():
    """Test edge cases."""
    result = ghosal_entropy_cnd(np.array([42.0]))
    assert result["n"] == 1
