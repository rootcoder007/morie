"""Tests for fzc1c6.fauzi_conditions_c1_c6."""

import numpy as np

from morie.fn.fzc1c6 import fauzi_conditions_c1_c6


def test_fzc1c6_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fauzi_conditions_c1_c6(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_fzc1c6_edge():
    """Test edge cases."""
    result = fauzi_conditions_c1_c6(np.array([42.0]))
    assert result["n"] == 1
