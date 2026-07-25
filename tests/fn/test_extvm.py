"""Tests for extvm.extreme_value_gev."""

import numpy as np

from morie.fn.extvm import extreme_value_gev


def test_extvm_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = extreme_value_gev(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_extvm_edge():
    """Test edge cases."""
    result = extreme_value_gev(np.array([42.0]))
    assert result["n"] == 1
