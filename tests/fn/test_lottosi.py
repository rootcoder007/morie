"""Tests for lottosi.lottery_sampling."""

from morie.fn import _array_core as np

from morie.fn.lottosi import lottery_sampling


def test_lottosi_basic():
    """Test basic functionality."""
    z = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    n = 5
    result = lottery_sampling(z, y, n)
    assert isinstance(result, dict)
    assert "estimate" in result or "index" in result


def test_lottosi_edge():
    """Test edge cases."""
    z = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    n = 5
    result = lottery_sampling(z, y, n)
    assert isinstance(result, dict)
