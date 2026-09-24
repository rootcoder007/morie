"""Tests for modlar.modularity_newman."""

from morie.fn import _array_core as np

from morie.fn.modlar import modularity_newman


def test_modlar_basic():
    """Test basic functionality."""
    A = 0.5
    communities = 0.5
    result = modularity_newman(A, communities)
    assert isinstance(result, dict)
    assert "estimate" in result or "Q" in result


def test_modlar_edge():
    """Test edge cases."""
    A = 0.5
    communities = 0.5
    result = modularity_newman(A, communities)
    assert isinstance(result, dict)
