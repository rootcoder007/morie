"""Tests for sgdmom.sgd_momentum."""

from morie.fn import _array_core as np

from morie.fn.sgdmom import sgd_momentum


def test_sgdmom_basic():
    """Test basic functionality."""
    g = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = sgd_momentum(g)
    assert isinstance(result, dict)
    assert "update" in result


def test_sgdmom_edge():
    """Test edge cases."""
    g = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = sgd_momentum(g)
    assert isinstance(result, dict)
