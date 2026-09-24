"""Tests for nesterv.nesterov_accelerated."""

from morie.fn import _array_core as np

from morie.fn.nesterv import nesterov_accelerated


def test_nesterv_basic():
    """Test basic functionality."""
    g = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = nesterov_accelerated(g)
    assert isinstance(result, dict)
    assert "update" in result


def test_nesterv_edge():
    """Test edge cases."""
    g = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = nesterov_accelerated(g)
    assert isinstance(result, dict)
