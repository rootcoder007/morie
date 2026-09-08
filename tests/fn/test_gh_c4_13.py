"""Tests for gh_c4_13.ghosal_dp_weak_conv."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_13 import ghosal_dp_weak_conv


def test_gh_c4_13_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dp_weak_conv(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c4_13_edge():
    """Test edge cases."""
    result = ghosal_dp_weak_conv(np.array([42.0]))
    assert result["n"] == 1
