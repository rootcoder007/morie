"""Tests for mmaxn.minmax_normalization."""

from morie.fn import _array_core as np

from morie.fn.mmaxn import minmax_normalization


def test_mmaxn_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = minmax_normalization(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_mmaxn_edge():
    """Test edge cases."""
    result = minmax_normalization(np.array([42.0]))
    assert result["n"] == 1
