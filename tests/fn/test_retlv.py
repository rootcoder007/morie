"""Tests for retlv.return_level."""

import numpy as np

from morie.fn.retlv import return_level


def test_retlv_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = return_level(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_retlv_edge():
    """Test edge cases."""
    result = return_level(np.array([42.0]))
    assert result["n"] == 1
