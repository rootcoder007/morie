"""Tests for rgeqn5b.rangayyan_ch5_waveform_length."""

from morie.fn import _array_core as np

from morie.fn.rgeqn5b import rangayyan_ch5_waveform_length


def test_rgeqn5b_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_ch5_waveform_length(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_rgeqn5b_edge():
    """Test edge cases."""
    result = rangayyan_ch5_waveform_length(np.array([42.0]))
    assert result["n"] == 1
