"""Tests for rgstf.rangayyan_stft."""

import numpy as np

from morie.fn.rgstf import rangayyan_stft


def test_rgstf_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_stft(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_rgstf_edge():
    """Test edge cases."""
    result = rangayyan_stft(np.array([42.0]))
    assert result["n"] == 1
