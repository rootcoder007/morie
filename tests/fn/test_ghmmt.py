"""Tests for ghmmt.ghosal_moment_matching."""

from morie.fn import _array_core as np

from morie.fn.ghmmt import ghosal_moment_matching


def test_ghmmt_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_moment_matching(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_ghmmt_edge():
    """Test edge cases."""
    result = ghosal_moment_matching(np.array([42.0]))
    assert result["n"] == 1
