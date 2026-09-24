"""Tests for nde.natural_direct_effect."""

from morie.fn import _array_core as np

from morie.fn.nde import natural_direct_effect


def test_nde_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    M = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = natural_direct_effect(X, M, Y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_nde_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    M = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = natural_direct_effect(X, M, Y)
    assert isinstance(result, dict)
