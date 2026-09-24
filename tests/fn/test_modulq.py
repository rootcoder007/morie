"""Tests for modulq.modularity_q."""

from morie.fn import _array_core as np

from morie.fn.modulq import modularity_q


def test_modulq_basic():
    """Test basic functionality."""
    G = 0.5
    communities = 0.5
    result = modularity_q(G, communities)
    assert isinstance(result, dict)
    assert "estimate" in result or "Q" in result


def test_modulq_edge():
    """Test edge cases."""
    G = 0.5
    communities = 0.5
    result = modularity_q(G, communities)
    assert isinstance(result, dict)
