"""Tests for km145.kamath_ch9_mmllm_autoregressive."""

from morie.fn import _array_core as np

from morie.fn.km145 import kamath_ch9_mmllm_autoregressive


def test_km145_basic():
    """Test basic functionality."""
    R = [0.5, 0.25, 0.5]
    I = None
    result = kamath_ch9_mmllm_autoregressive(R, I)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_km145_edge():
    """Test edge cases."""
    R = [0.5, 0.25, 0.5]
    I = None
    result = kamath_ch9_mmllm_autoregressive(R, I)
    assert isinstance(result, dict)
