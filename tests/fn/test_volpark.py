"""Tests for volpark.vol_parkinson_range."""

from morie.fn import _array_core as np

from morie.fn.volpark import vol_parkinson_range


def test_volpark_basic():
    """Test basic functionality."""
    high = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    low = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = vol_parkinson_range(high, low)
    assert isinstance(result, dict)
    assert "variance" in result


def test_volpark_edge():
    """Test edge cases."""
    high = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    low = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = vol_parkinson_range(high, low)
    assert isinstance(result, dict)
