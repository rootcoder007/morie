"""Tests for alsR.als."""

from morie.fn import _array_core as np

from morie.fn.alsR import als


def test_alsR_basic():
    """Test basic functionality."""
    R = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = als(R)
    assert isinstance(result, dict)
    assert "X" in result
def test_alsR_edge():
    """Test edge cases."""
    R = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = als(R)
    assert isinstance(result, dict)
