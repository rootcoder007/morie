"""Tests for wsmmcd.wasserman_mcdiarmid."""

from morie.fn import _array_core as np

from morie.fn.wsmmcd import wasserman_mcdiarmid


def test_wsmmcd_basic():
    """Test basic functionality."""
    t = 0.1
    c = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = wasserman_mcdiarmid(t, c)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_wsmmcd_edge():
    """Test edge cases."""
    t = 0.1
    c = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = wasserman_mcdiarmid(t, c)
    assert isinstance(result, dict)
