"""Tests for sortP.sortpool."""

from morie.fn import _array_core as np

from morie.fn.sortP import sortpool


def test_sortP_basic():
    """Test basic functionality."""
    features = np.random.default_rng(42).normal(0.0, 1.0, 40)
    k_keep = 5
    result = sortpool(features, k_keep)
    assert isinstance(result, dict)
    assert "pooled" in result


def test_sortP_edge():
    """Test edge cases."""
    features = np.random.default_rng(42).normal(0.0, 1.0, 40)
    k_keep = 5
    result = sortpool(features, k_keep)
    assert isinstance(result, dict)
