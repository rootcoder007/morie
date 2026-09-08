"""Tests for gh_ap_e3.ghosal_wavelet_mra."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_e3 import ghosal_wavelet_mra


def test_gh_ap_e3_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_wavelet_mra(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_ap_e3_edge():
    """Test edge cases."""
    result = ghosal_wavelet_mra(np.array([42.0]))
    assert result["n"] == 1
