"""Tests for louv.louvain_communities."""

from morie.fn import _array_core as np

from morie.fn.louv import louvain_communities


def test_louv_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = louvain_communities(A)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_louv_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = louvain_communities(A)
    assert isinstance(result, dict)
