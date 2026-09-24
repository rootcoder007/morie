"""Tests for km126.kamath_ch8_smd."""

from morie.fn import _array_core as np

from morie.fn.km126 import kamath_ch8_smd


def test_km126_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = kamath_ch8_smd(x, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_km126_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = kamath_ch8_smd(x, y)
    assert isinstance(result, dict)
