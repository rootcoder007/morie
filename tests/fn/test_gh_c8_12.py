"""Tests for gh_c8_12.ghosal_crt_lower."""

from morie.fn import _array_core as np

from morie.fn.gh_c8_12 import ghosal_crt_lower


def test_gh_c8_12_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_crt_lower(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c8_12_edge():
    """Test edge cases."""
    result = ghosal_crt_lower(np.array([42.0]))
    assert result["n"] == 1
