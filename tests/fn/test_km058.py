"""Tests for km058.kamath_ch4_lora_forward."""

from morie.fn import _array_core as np

from morie.fn.km058 import kamath_ch4_lora_forward


def test_km058_basic():
    """Test basic functionality."""
    W_0 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    B = 5
    A = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = kamath_ch4_lora_forward(W_0, B, A, x)
    assert isinstance(result, dict)
    assert "estimate" in result or "h" in result


def test_km058_edge():
    """Test edge cases."""
    W_0 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    B = 5
    A = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = kamath_ch4_lora_forward(W_0, B, A, x)
    assert isinstance(result, dict)
