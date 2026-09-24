"""Tests for hmgcl.geron_gradient_clipping."""

from morie.fn import _array_core as np

from morie.fn.hmgcl import geron_gradient_clipping


def test_hmgcl_basic():
    """Test basic functionality."""
    grads = np.random.default_rng(42).normal(0.0, 1.0, 40)
    max_norm = 0.1
    result = geron_gradient_clipping(grads, max_norm)
    assert isinstance(result, dict)
    assert "estimate" in result or "clipped" in result


def test_hmgcl_edge():
    """Test edge cases."""
    grads = np.random.default_rng(42).normal(0.0, 1.0, 40)
    max_norm = 0.1
    result = geron_gradient_clipping(grads, max_norm)
    assert isinstance(result, dict)
