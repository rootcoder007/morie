"""Tests for sgtbtw.sgt_betweenness_centrality."""

from morie.fn import _array_core as np

from morie.fn.sgtbtw import sgt_betweenness_centrality


def test_sgtbtw_basic():
    """Test basic functionality."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_betweenness_centrality(A)
    assert isinstance(result, dict)
    assert "betweenness" in result


def test_sgtbtw_edge():
    """Test edge cases."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_betweenness_centrality(A)
    assert isinstance(result, dict)
