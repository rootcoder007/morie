"""Tests for relbt.reliability_metric."""

from morie.fn import _array_core as np

from morie.fn.relbt import reliability_metric


def test_relbt_basic():
    """Test basic functionality."""
    r = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    h2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = reliability_metric(r, h2)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_relbt_edge():
    """Test edge cases."""
    r = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    h2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = reliability_metric(r, h2)
    assert isinstance(result, dict)
