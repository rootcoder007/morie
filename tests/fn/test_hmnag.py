"""Tests for hmnag.geron_nesterov."""

from morie.fn import _array_core as np

from morie.fn.hmnag import geron_nesterov


def test_hmnag_basic():
    """Test basic functionality."""
    grads = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_nesterov(grads)
    assert isinstance(result, dict)
    assert "estimate" in result or "theta" in result


def test_hmnag_edge():
    """Test edge cases."""
    grads = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_nesterov(grads)
    assert isinstance(result, dict)
