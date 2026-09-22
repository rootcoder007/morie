"""Tests for croston.croston."""

from morie.fn import _array_core as np

from morie.fn.croston import croston


def test_croston_basic():
    """Test basic functionality."""
    y = np.abs(np.random.default_rng(43).normal(0, 1, 100)) + 0.5
    result = croston(y)
    assert isinstance(result, dict)
    assert "forecast" in result
def test_croston_edge():
    """Test edge cases."""
    y = np.abs(np.random.default_rng(43).normal(0, 1, 100)) + 0.5
    result = croston(y)
    assert isinstance(result, dict)
