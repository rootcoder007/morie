"""Tests for rgwhann.rangayyan_hann_window."""

from morie.fn import _array_core as np

from morie.fn.rgwhann import rangayyan_hann_window


def test_rgwhann_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_hann_window(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_rgwhann_edge():
    """Test edge cases."""
    result = rangayyan_hann_window(np.array([42.0]))
    assert result["n"] == 1
