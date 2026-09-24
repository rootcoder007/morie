"""Tests for netdeg.degree_centrality."""

from morie.fn import _array_core as np

from morie.fn.netdeg import degree_centrality


def test_netdeg_basic():
    """Test basic functionality."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = degree_centrality(A)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_netdeg_edge():
    """Test edge cases."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = degree_centrality(A)
    assert isinstance(result, dict)
