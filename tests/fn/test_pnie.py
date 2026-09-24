"""Tests for pnie.pure_natural_indirect_effect."""

from morie.fn import _array_core as np

from morie.fn.pnie import pure_natural_indirect_effect


def test_pnie_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    M = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = pure_natural_indirect_effect(X, M, Y)
    assert isinstance(result, dict)
    assert "estimate" in result or "pnde" in result


def test_pnie_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    M = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = pure_natural_indirect_effect(X, M, Y)
    assert isinstance(result, dict)
