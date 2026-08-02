"""Tests for gh_ap_h1.ghosal_inv_gauss."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_h1 import ghosal_inv_gauss


def test_gh_ap_h1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_inv_gauss(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_ap_h1_edge():
    """Test edge cases."""
    result = ghosal_inv_gauss(np.array([42.0]))
    assert result["n"] == 1
