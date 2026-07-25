"""Tests for gb_ar7.gibbons_are_logistic."""

import numpy as np

from morie.fn.gb_ar7 import gibbons_are_logistic


def test_gb_ar7_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gibbons_are_logistic(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gb_ar7_edge():
    """Test edge cases."""
    result = gibbons_are_logistic(np.array([42.0]))
    assert result["n"] == 1
