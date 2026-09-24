"""Tests for bndmnt.bound_monotone_test."""

import pytest

from morie.fn import _array_core as np

from morie.fn.bndmnt import bound_monotone_test


def test_bndmnt_basic():
    """Test basic functionality raises NotImplementedError."""
    rng = np.random.default_rng(42)
    n = 40
    p = 3
    y = rng.normal(0, 1, n)
    D = rng.integers(0, 2, n)
    X = rng.normal(0, 1, (n, p))
    with pytest.raises(NotImplementedError):
        bound_monotone_test(y, D, X)


def test_bndmnt_edge():
    """Test edge cases raise NotImplementedError."""
    rng = np.random.default_rng(7)
    n = 40
    p = 3
    y = rng.normal(0, 1, n)
    D = rng.integers(0, 2, n)
    X = rng.normal(0, 1, (n, p))
    with pytest.raises(NotImplementedError):
        bound_monotone_test(y, D, X)
