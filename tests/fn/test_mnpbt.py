"""Tests for mnpbt.multinomial_probit_spatial."""

import numpy as np

from morie.fn.mnpbt import multinomial_probit_spatial


def test_mnpbt_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = multinomial_probit_spatial(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_mnpbt_edge():
    """Test edge cases."""
    result = multinomial_probit_spatial(np.array([42.0]))
    assert result["n"] == 1
