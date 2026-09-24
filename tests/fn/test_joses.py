"""Tests for joses.joseph_simple_exponential_smoothing."""

from morie.fn import _array_core as np

from morie.fn.joses import joseph_simple_exponential_smoothing


def test_joses_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_simple_exponential_smoothing(y)
    assert isinstance(result, dict)
    assert "forecast" in result


def test_joses_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_simple_exponential_smoothing(y)
    assert isinstance(result, dict)
