"""Tests for merck.mercer_theorem."""

import numpy as np

from morie.fn.merck import mercer_theorem


def test_merck_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = mercer_theorem(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_merck_edge():
    """Test edge cases."""
    result = mercer_theorem(np.array([42.0]))
    assert result["n"] == 1
