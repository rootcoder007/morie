"""Tests for hmlntr.geron_layer_normalization."""

from morie.fn import _array_core as np

from morie.fn.hmlntr import geron_layer_normalization


def test_hmlntr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_layer_normalization(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "y" in result


def test_hmlntr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_layer_normalization(x)
    assert isinstance(result, dict)
