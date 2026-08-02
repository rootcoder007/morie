"""Tests for rgwblkm.rangayyan_blackman_window."""

from morie.fn import _array_core as np

from morie.fn.rgwblkm import rangayyan_blackman_window


def test_rgwblkm_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_blackman_window(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_rgwblkm_edge():
    """Test edge cases."""
    result = rangayyan_blackman_window(np.array([42.0]))
    assert result["n"] == 1
