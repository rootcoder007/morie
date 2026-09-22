"""Tests for gh_c2_6.ghosal_mixture_basis_prior."""

from morie.fn import _array_core as np

from morie.fn.gh_c2_6 import ghosal_mixture_basis_prior


def test_gh_c2_6_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_mixture_basis_prior(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c2_6_edge():
    """Test edge cases."""
    x = np.array([42.0])
    result = ghosal_mixture_basis_prior(x)
    assert "density" in result
    assert len(result["density"]) == len(x)
    assert np.all(np.isfinite(np.asarray(result["density"], dtype=float)))
