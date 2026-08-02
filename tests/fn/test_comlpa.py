"""Tests for comlpa.label_propagation."""

from morie.fn import _array_core as np

from morie.fn.comlpa import label_propagation


def test_comlpa_basic():
    """Test basic functionality."""
    G = np.eye(10)
    result = label_propagation(G)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_comlpa_edge():
    """Test edge cases."""
    G = np.eye(10)
    result = label_propagation(G)
    assert isinstance(result, dict)
