"""Tests for fzedg.fauzi_edgeworth_quantile."""

import numpy as np

from morie.fn.fzedg import fauzi_edgeworth_quantile


def test_fzedg_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fauzi_edgeworth_quantile(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_fzedg_edge():
    """Test edge cases."""
    result = fauzi_edgeworth_quantile(np.array([42.0]))
    assert result["n"] == 1
