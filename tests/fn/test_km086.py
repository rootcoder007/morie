"""Tests for km086.kamath_ch6_pll."""

from morie.fn import _array_core as np

from morie.fn.km086 import kamath_ch6_pll


def test_km086_basic():
    """Test basic functionality."""
    S = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_ch6_pll(S)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_km086_edge():
    """Test edge cases."""
    S = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_ch6_pll(S)
    assert isinstance(result, dict)
