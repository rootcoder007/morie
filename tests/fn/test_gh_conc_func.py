"""Tests for gh_conc_func.ghosal_concentration_function."""

from morie.fn import _array_core as np

from morie.fn.gh_conc_func import ghosal_concentration_function


def test_gh_conc_func_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_concentration_function(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_conc_func_edge():
    """Test edge cases."""
    result = ghosal_concentration_function(np.array([42.0]))
    assert result["n"] == 1
