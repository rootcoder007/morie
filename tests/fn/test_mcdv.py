"""Tests for mcdv.mcd."""

from morie.fn import _array_core as np

from morie.fn.mcdv import mcd


def test_mcdv_basic():
    """Test basic functionality."""
    X = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = mcd(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_mcdv_edge():
    """Test edge cases."""
    X = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = mcd(X)
    assert isinstance(result, dict)
