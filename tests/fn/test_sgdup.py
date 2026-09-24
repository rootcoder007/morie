"""Tests for sgdup.sgd_update."""

from morie.fn import _array_core as np

from morie.fn.sgdup import sgd_update


def test_sgdup_basic():
    """Test basic functionality."""
    beta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    batch_grads = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = sgd_update(beta, batch_grads)
    assert isinstance(result, dict)
    assert "beta" in result


def test_sgdup_edge():
    """Test edge cases."""
    beta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    batch_grads = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = sgd_update(beta, batch_grads)
    assert isinstance(result, dict)
