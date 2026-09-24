"""Tests for netbtw.betweenness_centrality."""

from morie.fn import _array_core as np

from morie.fn.netbtw import betweenness_centrality


def test_netbtw_basic():
    """Test basic functionality."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = betweenness_centrality(A)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_netbtw_edge():
    """Test edge cases."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = betweenness_centrality(A)
    assert isinstance(result, dict)
