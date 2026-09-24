"""Tests for hmmom.geron_momentum."""

from morie.fn import _array_core as np

from morie.fn.hmmom import geron_momentum


def test_hmmom_basic():
    """Test basic functionality."""
    grads = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_momentum(grads)
    assert isinstance(result, dict)
    assert "estimate" in result or "theta" in result


def test_hmmom_edge():
    """Test edge cases."""
    grads = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_momentum(grads)
    assert isinstance(result, dict)
