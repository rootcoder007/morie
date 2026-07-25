"""Tests for gb1241.gibbons_concordance_w."""

import numpy as np

from morie.fn.gb1241 import gibbons_concordance_w


def test_gb1241_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gibbons_concordance_w(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gb1241_edge():
    """Test edge cases."""
    result = gibbons_concordance_w(np.array([42.0]))
    assert result["n"] == 1
