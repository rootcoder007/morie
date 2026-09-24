"""Tests for hmrmsp.geron_rmsprop."""

from morie.fn import _array_core as np

from morie.fn.hmrmsp import geron_rmsprop


def test_hmrmsp_basic():
    """Test basic functionality."""
    grads = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_rmsprop(grads)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_hmrmsp_edge():
    """Test edge cases."""
    grads = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_rmsprop(grads)
    assert isinstance(result, dict)
