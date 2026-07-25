"""Tests for dccmd.dcc_multivariate_garch."""

import numpy as np

from morie.fn.dccmd import dcc_multivariate_garch


def test_dccmd_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = dcc_multivariate_garch(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_dccmd_edge():
    """Test edge cases."""
    result = dcc_multivariate_garch(np.array([42.0]))
    assert result["n"] == 1
