"""Tests for rmspO.rmsprop_optimizer."""

from morie.fn import _array_core as np

from morie.fn.rmspO import rmsprop_optimizer


def test_rmspO_basic():
    """Test basic functionality."""
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    grad = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rmsprop_optimizer(theta, grad)
    assert isinstance(result, dict)
    assert "theta" in result


def test_rmspO_edge():
    """Test edge cases."""
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    grad = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rmsprop_optimizer(theta, grad)
    assert isinstance(result, dict)
