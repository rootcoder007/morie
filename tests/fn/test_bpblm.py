"""Tests for bpblm.bits_per_byte."""

import numpy as np

from morie.fn.bpblm import bits_per_byte


def test_bpblm_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = bits_per_byte(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_bpblm_edge():
    """Test edge cases."""
    result = bits_per_byte(np.array([42.0]))
    assert result["n"] == 1
