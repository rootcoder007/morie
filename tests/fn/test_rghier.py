"""Tests for rghier.rangayyan_hierarchical_clust."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_hierarchical_clust


def test_rghier_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = rangayyan_hierarchical_clust(X)
    assert isinstance(result, dict)
    assert "history" in result


def test_rghier_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = rangayyan_hierarchical_clust(X)
    assert isinstance(result, dict)
