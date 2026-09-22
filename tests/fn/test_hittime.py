"""Tests for hittime.hitting_time."""

from morie.fn import _array_core as np

from morie.fn.hittime import hitting_time


def test_hittime_basic():
    """Test basic functionality."""
    G = np.eye(10)
    result = hitting_time(G)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hittime_edge():
    """Test edge cases."""
    G = np.eye(10)
    result = hitting_time(G)
    assert isinstance(result, dict)
