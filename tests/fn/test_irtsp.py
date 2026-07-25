"""Tests for irtsp.irt_spatial."""

import numpy as np

from morie.fn.irtsp import irt_spatial


def test_irtsp_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = irt_spatial(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_irtsp_edge():
    """Test edge cases."""
    result = irt_spatial(np.array([42.0]))
    assert result["n"] == 1
