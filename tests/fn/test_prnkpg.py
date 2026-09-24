"""Tests for prnkpg.pagerank."""

from morie.fn import _array_core as np

from morie.fn.prnkpg import pagerank


def test_prnkpg_basic():
    """Test basic functionality."""
    G = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = pagerank(G)
    assert isinstance(result, dict)
    assert "estimate" in result or "pr" in result


def test_prnkpg_edge():
    """Test edge cases."""
    G = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = pagerank(G)
    assert isinstance(result, dict)
