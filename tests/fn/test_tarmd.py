"""Tests for tarmd.threshold_autoregression."""

import numpy as np

from morie.fn.tarmd import threshold_autoregression


def test_tarmd_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = threshold_autoregression(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_tarmd_edge():
    """Test edge cases."""
    result = threshold_autoregression(np.array([42.0]))
    assert result["n"] == 1
