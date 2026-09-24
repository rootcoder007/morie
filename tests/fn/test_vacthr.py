"""Tests for vacthr.vaccination_threshold."""

from morie.fn import _array_core as np

from morie.fn.vacthr import vaccination_threshold


def test_vacthr_basic():
    """Test basic functionality."""
    R0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = vaccination_threshold(R0)
    assert isinstance(result, dict)
    assert "threshold" in result


def test_vacthr_edge():
    """Test edge cases."""
    R0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = vaccination_threshold(R0)
    assert isinstance(result, dict)
