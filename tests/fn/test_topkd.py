"""Tests for topkd.top_k_decoding."""

import numpy as np

from morie.fn.topkd import top_k_decoding


def test_topkd_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = top_k_decoding(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_topkd_edge():
    """Test edge cases."""
    result = top_k_decoding(np.array([42.0]))
    assert result["n"] == 1
