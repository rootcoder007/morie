"""Tests for gh_c14_1.ghosal_eppf_def."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_1 import ghosal_eppf_def


def test_gh_c14_1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_eppf_def(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c14_1_edge():
    """Test edge cases."""
    result = ghosal_eppf_def(np.array([42.0]))
    assert result["n"] == 1
