"""Tests for posab.positional_encoding_abs."""

import numpy as np

from morie.fn.posab import positional_encoding_abs


def test_posab_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = positional_encoding_abs(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_posab_edge():
    """Test edge cases."""
    result = positional_encoding_abs(np.array([42.0]))
    assert result["n"] == 1
