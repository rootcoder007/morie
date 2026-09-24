"""Tests for rgnmf.rangayyan_nmf."""

import math

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_nmf


def test_rgnmf_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p, r = 40, 20, 3
    V = np.abs(rng.normal(0, 1, (n, p)))
    maxiter = 100
    tol = 1e-6
    result = rangayyan_nmf(V, r, maxiter, tol)
    assert isinstance(result, dict)


def test_rgnmf_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p, r = 10, 5, 1
    V = np.abs(rng.normal(0, 1, (n, p)))
    maxiter = 50
    tol = 1e-6
    result = rangayyan_nmf(V, r, maxiter, tol)
    assert isinstance(result, dict)
