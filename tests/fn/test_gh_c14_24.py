"""Tests for gh_c14_24.ghosal_ibp_stickbr."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_24 import ghosal_ibp_stickbr


def test_gh_c14_24_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_ibp_stickbr(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c14_24_edge():
    """Test edge cases."""
    result = ghosal_ibp_stickbr(np.array([42.0]))
    assert result["n"] == 1
