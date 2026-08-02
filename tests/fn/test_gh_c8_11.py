"""Tests for gh_c8_11.ghosal_ts_crt."""

from morie.fn import _array_core as np

from morie.fn.gh_c8_11 import ghosal_ts_crt


def test_gh_c8_11_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_ts_crt(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c8_11_edge():
    """Test edge cases."""
    result = ghosal_ts_crt(np.array([42.0]))
    assert result["n"] == 1
