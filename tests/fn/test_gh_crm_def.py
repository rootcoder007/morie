"""Tests for gh_crm_def.ghosal_completely_random_measure."""

from morie.fn import _array_core as np

from morie.fn.gh_crm_def import ghosal_completely_random_measure


def test_gh_crm_def_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_completely_random_measure(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_crm_def_edge():
    """Test edge cases."""
    result = ghosal_completely_random_measure(np.array([42.0]))
    assert result["n"] == 1
