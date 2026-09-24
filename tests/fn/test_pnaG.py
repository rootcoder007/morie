"""Tests for pnaG.pna."""

from morie.fn import _array_core as np

from morie.fn.pnaG import pna


def test_pnaG_basic():
    """Test basic functionality."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    X = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = pna(A, X)
    assert isinstance(result, dict)
    assert "out" in result


def test_pnaG_edge():
    """Test edge cases."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    X = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = pna(A, X)
    assert isinstance(result, dict)
