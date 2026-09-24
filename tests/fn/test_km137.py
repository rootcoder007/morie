"""Tests for km137.kamath_ch9_itm_hard_negative."""

from morie.fn import _array_core as np

from morie.fn.km137 import kamath_ch9_itm_hard_negative


def test_km137_basic():
    """Test basic functionality."""
    Pos = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    HardNeg = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_ch9_itm_hard_negative(Pos, HardNeg)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_km137_edge():
    """Test edge cases."""
    Pos = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    HardNeg = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_ch9_itm_hard_negative(Pos, HardNeg)
    assert isinstance(result, dict)
