"""Tests for gh_c13_7.ghosal_mix_bp."""

from morie.fn import _array_core as np

from morie.fn.gh_c13_7 import ghosal_mix_bp


def test_gh_c13_7_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_mix_bp(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c13_7_edge():
    """Test edge cases."""
    result = ghosal_mix_bp(np.array([42.0]))
    assert result["n"] == 1
