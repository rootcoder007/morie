"""Tests for reniyd.renyi_entropy."""

from morie.fn import _array_core as np

from morie.fn.reniyd import renyi_entropy


def test_reniyd_basic():
    """Test basic functionality."""
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = renyi_entropy(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_reniyd_edge():
    """Test edge cases."""
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = renyi_entropy(y)
    assert isinstance(result, dict)
