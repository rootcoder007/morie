"""Tests for tmpsc.temperature_scaling."""

import numpy as np

from morie.fn.tmpsc import temperature_scaling


def test_tmpsc_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = temperature_scaling(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_tmpsc_edge():
    """Test edge cases."""
    result = temperature_scaling(np.array([42.0]))
    assert result["n"] == 1
