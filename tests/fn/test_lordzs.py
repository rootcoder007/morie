"""Tests for lordzs.lord_chi_square."""

from morie.fn import _array_core as np

from morie.fn.lordzs import lord_chi_square


def test_lordzs_basic():
    """Test basic functionality."""
    b_R = 0.5
    b_F = 0.5
    V_R = 0.5
    result = lord_chi_square(b_R, b_F, V_R)
    assert isinstance(result, dict)
    assert "statistic" in result or "statistic" in result or "statistic" in result


def test_lordzs_edge():
    """Test edge cases."""
    b_R = 0.5
    b_F = 0.5
    V_R = 0.5
    result = lord_chi_square(b_R, b_F, V_R)
    assert isinstance(result, dict)
