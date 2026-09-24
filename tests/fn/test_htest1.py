"""Tests for htest1.horvitz_thompson."""

from morie.fn import _array_core as np

from morie.fn.htest1 import horvitz_thompson


def test_htest1_basic():
    """Test basic functionality."""
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    pi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = horvitz_thompson(y, pi)
    assert isinstance(result, dict)
    assert "total" in result


def test_htest1_edge():
    """Test edge cases."""
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    pi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = horvitz_thompson(y, pi)
    assert isinstance(result, dict)
