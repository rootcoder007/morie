"""Tests for gh_mises_eff.ghosal_mises_efficiency."""

from morie.fn import _array_core as np

from morie.fn.gh_mises_eff import ghosal_mises_efficiency


def test_gh_mises_eff_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_mises_efficiency(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_mises_eff_edge():
    """Test edge cases."""
    result = ghosal_mises_efficiency(np.array([42.0]))
    assert result["n"] == 1
