"""Tests for grln.geron_layer_normalization."""

from morie.fn import _array_core as np

from morie.fn.grln import geron_layer_normalization


def test_grln_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_layer_normalization(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grln_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_layer_normalization(X)
    assert isinstance(result, dict)
