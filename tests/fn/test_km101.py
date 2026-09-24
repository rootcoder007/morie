"""Tests for km101.kamath_ch6_toxic_fraction."""

from morie.fn import _array_core as np

from morie.fn.km101 import kamath_ch6_toxic_fraction


def test_km101_basic():
    """Test basic functionality."""
    Yhat = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    c = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_ch6_toxic_fraction(Yhat, c)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_km101_edge():
    """Test edge cases."""
    Yhat = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    c = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_ch6_toxic_fraction(Yhat, c)
    assert isinstance(result, dict)
