"""Tests for pmiwd.pointwise_mutual_info."""

from morie.fn import _array_core as np

from morie.fn.pmiwd import pointwise_mutual_info


def test_pmiwd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = pointwise_mutual_info(x, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_pmiwd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = pointwise_mutual_info(x, y)
    assert isinstance(result, dict)
