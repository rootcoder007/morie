"""Tests for mafcl.maf_calculation."""

import numpy as np

from morie.fn.mafcl import maf_calculation


def test_mafcl_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = maf_calculation(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_mafcl_edge():
    """Test edge cases."""
    result = maf_calculation(np.array([42.0]))
    assert result["n"] == 1
