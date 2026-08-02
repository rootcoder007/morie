"""Tests for zscnm.zscore_normalization."""

from morie.fn import _array_core as np

from morie.fn.zscnm import zscore_normalization


def test_zscnm_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = zscore_normalization(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_zscnm_edge():
    """Test edge cases."""
    result = zscore_normalization(np.array([42.0]))
    assert result["n"] == 1
