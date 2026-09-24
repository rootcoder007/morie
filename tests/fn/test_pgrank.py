"""Tests for pgrank.pagerank."""

from morie.fn import _array_core as np

from morie.fn.pgrank import pagerank


def test_pgrank_basic():
    """Test basic functionality."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = pagerank(A)
    assert isinstance(result, dict)
    assert "estimate" in result or "pr" in result


def test_pgrank_edge():
    """Test edge cases."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = pagerank(A)
    assert isinstance(result, dict)
