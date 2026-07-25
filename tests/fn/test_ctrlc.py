"""Tests for ctrlc.control_comparison."""

import numpy as np

from morie.fn.ctrlc import control_comparison


def test_ctrlc_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = control_comparison(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_ctrlc_edge():
    """Test edge cases."""
    result = control_comparison(np.array([42.0]))
    assert result["n"] == 1
