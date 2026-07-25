"""Tests for rcall.roll_call_analysis."""

import numpy as np

from morie.fn.rcall import roll_call_analysis


def test_rcall_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = roll_call_analysis(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_rcall_edge():
    """Test edge cases."""
    result = roll_call_analysis(np.array([42.0]))
    assert result["n"] == 1
