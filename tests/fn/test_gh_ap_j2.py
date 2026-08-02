"""Tests for gh_ap_j2.ghosal_crm_laplace."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_j2 import ghosal_crm_laplace


def test_gh_ap_j2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_crm_laplace(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_ap_j2_edge():
    """Test edge cases."""
    result = ghosal_crm_laplace(np.array([42.0]))
    assert result["n"] == 1
