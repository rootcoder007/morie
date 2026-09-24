"""Tests for macn.ma_cochran_q."""

from morie.fn import _array_core as np

from morie.fn.macn import ma_cochran_q


def test_macn_basic():
    """Test basic functionality."""
    yi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    vi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ma_cochran_q(yi, vi)
    assert isinstance(result, dict)
    assert "statistic" in result or "statistic" in result or "statistic" in result


def test_macn_edge():
    """Test edge cases."""
    yi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    vi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ma_cochran_q(yi, vi)
    assert isinstance(result, dict)
