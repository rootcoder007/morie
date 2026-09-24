"""Tests for mscmcl.matrix_completion_scm."""

import math

from morie.fn import _array_core as np

from morie.fn.mscmcl import matrix_completion_scm


def test_mscmcl_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_d = np.random.default_rng(42)
    N, T = 20, 8
    y = rng_y.normal(0, 1, (N, T))
    D = rng_d.integers(0, 2, (N, T))
    lam = 0.1
    result = matrix_completion_scm(y, D, lam)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "att" in result
    assert "L" in result
    assert "tau" in result
    assert "n_treated" in result
    assert "rank" in result
    assert "nuclear" in result
    assert "iterations" in result
    assert "converged" in result
    assert "N" in result
    assert "T" in result
    assert result["N"] == N
    assert result["T"] == T
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["att"])
    assert len(result["L"]) == N


def test_mscmcl_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    rng_d = np.random.default_rng(42)
    N, T = 15, 6
    y = rng_y.normal(0, 1, (N, T))
    D = rng_d.integers(0, 2, (N, T))
    # lam = 0 is the degenerate limit explicitly described in the docstring;
    # the projection step is still well defined, so this is a valid input.
    lam = 0.0
    result = matrix_completion_scm(y, D, lam)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "att" in result
    assert "converged" in result
    assert "iterations" in result
    assert "N" in result
    assert "T" in result
    assert result["N"] == N
    assert result["T"] == T
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["iterations"])
