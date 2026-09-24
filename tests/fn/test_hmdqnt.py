"""Tests for hmdqnt.geron_dynamic_quantization."""

from morie.fn import _array_core as np

from morie.fn.hmdqnt import geron_dynamic_quantization


def test_hmdqnt_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_dynamic_quantization(model)
    assert isinstance(result, dict)
    assert "estimate" in result or "quantized" in result


def test_hmdqnt_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_dynamic_quantization(model)
    assert isinstance(result, dict)
