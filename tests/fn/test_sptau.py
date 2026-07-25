"""Tests for sptau.spatial_autocorrelation."""

import numpy as np

from morie.fn.sptau import spatial_autocorrelation


def test_sptau_basic():
    """Test basic functionality."""
    x = np.arange(10, dtype=float)
    y = x * 2 + 1
    result = spatial_autocorrelation(x, y)
    assert np.all(np.isfinite(np.asarray(result["statistic"], dtype=float)))  # N6: was a generator-guessed value


def test_sptau_edge():
    """Test edge cases."""
    result = spatial_autocorrelation(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert result["n"] == 2
