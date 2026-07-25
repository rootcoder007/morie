"""Tests for heinz.he_initialization."""

import numpy as np

from morie.fn.heinz import he_initialization


def test_heinz_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = he_initialization(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_heinz_edge():
    """Test edge cases."""
    result = he_initialization(np.array([42.0]))
    assert result["n"] == 1
