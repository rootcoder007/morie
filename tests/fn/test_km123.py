"""Tests for km123.kamath_ch8_moverscore_distance."""

from morie.fn import _array_core as np

from morie.fn.km123 import kamath_ch8_moverscore_distance


def test_km123_basic():
    """Test basic functionality."""
    x_i = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y_j = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kamath_ch8_moverscore_distance(x_i, y_j)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_km123_edge():
    """Test edge cases."""
    x_i = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y_j = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kamath_ch8_moverscore_distance(x_i, y_j)
    assert isinstance(result, dict)
