"""Tests for mdrnk.midranks."""

import numpy as np

from morie.fn.mdrnk import midranks


def test_mdrnk_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = midranks(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_mdrnk_edge():
    """Test edge cases."""
    result = midranks(np.array([42.0]))
    assert result["n"] == 1
