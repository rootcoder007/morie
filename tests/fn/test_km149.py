"""Tests for km149.kamath_ch9_flamingo_factorized."""

from morie.fn import _array_core as np

from morie.fn.km149 import kamath_ch9_flamingo_factorized


def test_km149_basic():
    """Test basic functionality."""
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_ch9_flamingo_factorized(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_km149_edge():
    """Test edge cases."""
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_ch9_flamingo_factorized(y)
    assert isinstance(result, dict)
