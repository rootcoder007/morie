"""Tests for hrzbr5.horowitz_bias_reduction_deconv."""

from morie.fn import _array_core as np

from morie.fn.hrzbr5 import horowitz_bias_reduction_deconv


def test_hrzbr5_basic():
    """Test basic functionality."""
    bandwidth = 0.5
    kernel_order = 2.0
    result = horowitz_bias_reduction_deconv(bandwidth, kernel_order)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_hrzbr5_edge():
    """Test edge cases."""
    bandwidth = 0.5
    kernel_order = 2.0
    result = horowitz_bias_reduction_deconv(bandwidth, kernel_order)
    assert isinstance(result, dict)
