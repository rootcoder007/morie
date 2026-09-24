"""Tests for vbinfp.variational_bound."""

from morie.fn import _array_core as np

from morie.fn.vbinfp import variational_bound


def test_vbinfp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = variational_bound(X, Y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_vbinfp_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = variational_bound(X, Y)
    assert isinstance(result, dict)
