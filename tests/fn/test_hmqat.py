"""Tests for hmqat.geron_quantization_aware_training."""

from morie.fn import _array_core as np

from morie.fn.hmqat import geron_quantization_aware_training


def test_hmqat_basic():
    """Test basic functionality."""
    model = 0.5
    X = 0.5
    y = 0.5
    result = geron_quantization_aware_training(model, X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "weights" in result


def test_hmqat_edge():
    """Test edge cases."""
    model = 0.5
    X = 0.5
    y = 0.5
    result = geron_quantization_aware_training(model, X, y)
    assert isinstance(result, dict)
