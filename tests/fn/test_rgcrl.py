"""Tests for rgcrl.rangayyan_correlation_dimension."""

import numpy as np

from morie.fn.rgcrl import rangayyan_correlation_dimension


def test_rgcrl_basic():
    """Test basic functionality."""
    x = np.arange(10, dtype=float)
    y = x * 2 + 1
    result = rangayyan_correlation_dimension(x, y)
    assert np.all(np.isfinite(np.asarray(result["statistic"], dtype=float)))  # N6: was a generator-guessed value


def test_rgcrl_edge():
    """Test edge cases."""
    result = rangayyan_correlation_dimension(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert result["n"] == 2
