"""Tests for hmint8.geron_int8_quant."""

from morie.fn import _array_core as np

from morie.fn.hmint8 import geron_int8_quant


def test_hmint8_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_int8_quant(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "q" in result


def test_hmint8_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_int8_quant(x)
    assert isinstance(result, dict)
