"""Tests for gh_c14_5.ghosal_ssp_def."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_5 import ghosal_ssp_def


def test_gh_c14_5_basic():
    """Test basic functionality."""
    weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    atoms = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = sum(pi * t for pi, t in zip(weights, atoms))
    result = ghosal_ssp_def(weights, atoms)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert np.isclose(result["estimate"], expected)
    assert np.isclose(result["total_mass"], sum(weights))
    assert result["n_species"] == len(weights)
    assert "method" in result


def test_gh_c14_5_edge():
    """Test edge cases."""
    weights = np.array([1.0])
    atoms = np.array([42.0])
    result = ghosal_ssp_def(weights, atoms)
    assert result["n_species"] == 1
    assert np.isclose(result["estimate"], 42.0)
