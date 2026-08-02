"""Tests for gh_c11_10.ghosal_series_gp."""

from morie.fn import _array_core as np

from morie.fn.gh_c11_10 import ghosal_series_gp


def test_gh_c11_10_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_series_gp(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c11_10_edge():
    """Test edge cases."""
    result = ghosal_series_gp(np.array([42.0]))
    assert result["n"] == 1
