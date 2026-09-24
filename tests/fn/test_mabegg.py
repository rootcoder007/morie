"""Tests for mabegg.ma_begg_test."""

from morie.fn import _array_core as np

from morie.fn.mabegg import ma_begg_test


def test_mabegg_basic():
    """Test basic functionality."""
    yi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    vi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ma_begg_test(yi, vi)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "tau" in result


def test_mabegg_edge():
    """Test edge cases."""
    yi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    vi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ma_begg_test(yi, vi)
    assert isinstance(result, dict)
