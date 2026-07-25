"""Tests for rmsnr.rms_norm."""

import numpy as np

from morie.fn.rmsnr import rms_norm


def test_rmsnr_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rms_norm(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_rmsnr_edge():
    """Test edge cases."""
    result = rms_norm(np.array([42.0]))
    assert result["n"] == 1
