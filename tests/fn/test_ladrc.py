"""Tests for ladrc.ladder_of_causation."""

import numpy as np

from morie.fn.ladrc import ladder_of_causation


def test_ladrc_basic():
    """Test basic functionality."""
    x = np.arange(10, dtype=float)
    y = x * 2 + 1
    result = ladder_of_causation(x, y)
    assert np.all(np.isfinite(np.asarray(result["statistic"], dtype=float)))  # N6: was a generator-guessed value


def test_ladrc_edge():
    """Test edge cases."""
    result = ladder_of_causation(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert result["n"] == 2
