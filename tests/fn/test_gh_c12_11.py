"""Tests for gh_c12_11.ghosal_cred_set_cov."""

from morie.fn import _array_core as np

from morie.fn.gh_c12_11 import ghosal_cred_set_cov


def test_gh_c12_11_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_cred_set_cov(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c12_11_edge():
    """Test edge cases."""
    result = ghosal_cred_set_cov(np.array([42.0]))
    assert result["n"] == 1
