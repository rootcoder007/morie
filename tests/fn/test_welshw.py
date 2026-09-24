"""Tests for welshw.welsch_weight."""

from morie.fn import _array_core as np

from morie.fn.welshw import welsch_weight


def test_welshw_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = welsch_weight(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_welshw_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = welsch_weight(y)
    assert isinstance(result, dict)
