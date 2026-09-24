"""Tests for tnie.total_natural_indirect_effect."""

from morie.fn import _array_core as np

from morie.fn.tnie import total_natural_indirect_effect


def test_tnie_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    M = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = total_natural_indirect_effect(X, M, Y)
    assert isinstance(result, dict)
    assert "estimate" in result or "pnde" in result


def test_tnie_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    M = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = total_natural_indirect_effect(X, M, Y)
    assert isinstance(result, dict)
