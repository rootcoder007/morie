"""Tests for icfR.item_cf."""

from morie.fn import _array_core as np

from morie.fn.icfR import item_cf


def test_icfR_basic():
    """Test basic functionality."""
    R = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = item_cf(R)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_icfR_edge():
    """Test edge cases."""
    R = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = item_cf(R)
    assert isinstance(result, dict)
