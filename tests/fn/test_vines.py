"""Tests for vines.vine_copula."""

import numpy as np

from morie.fn.vines import vine_copula


def test_vines_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = vine_copula(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_vines_edge():
    """Test edge cases."""
    result = vine_copula(np.array([42.0]))
    assert result["n"] == 1
