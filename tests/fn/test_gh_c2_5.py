"""Tests for gh_c2_5.ghosal_histogram_prior."""

from morie.fn import _array_core as np

from morie.fn.gh_c2_5 import ghosal_histogram_prior


def test_gh_c2_5_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_histogram_prior(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c2_5_edge():
    """Test edge cases."""
    result = ghosal_histogram_prior(np.array([42.0]))
    assert result["n"] == 1
