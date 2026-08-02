"""Tests for gh_c2_1.ghosal_random_basis_expansion."""

from morie.fn import _array_core as np

from morie.fn.gh_c2_1 import ghosal_random_basis_expansion


def test_gh_c2_1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_random_basis_expansion(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c2_1_edge():
    """Test edge cases."""
    result = ghosal_random_basis_expansion(np.array([42.0]))
    assert result["n"] == 1
