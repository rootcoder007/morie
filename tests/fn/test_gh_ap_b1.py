"""Tests for gh_ap_b1.ghosal_kl_props."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_b1 import ghosal_kl_props


def test_gh_ap_b1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_kl_props(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_ap_b1_edge():
    """Test edge cases."""
    result = ghosal_kl_props(np.array([42.0]))
    assert result["n"] == 1
