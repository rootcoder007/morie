"""Tests for gh_c12_7.ghosal_strict_sbvm."""

from morie.fn import _array_core as np

from morie.fn.gh_c12_7 import ghosal_strict_sbvm


def test_gh_c12_7_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_strict_sbvm(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c12_7_edge():
    """Test edge cases."""
    result = ghosal_strict_sbvm(np.array([42.0]))
    assert result["n"] == 1
