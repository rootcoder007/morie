"""Tests for clusca.clustering_coefficient."""

from morie.fn import _array_core as np

from morie.fn.clusca import clustering_coefficient


def test_clusca_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    node = np.random.default_rng(42).normal(0, 1, 100)
    result = clustering_coefficient(y, A, node)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_clusca_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    node = np.random.default_rng(42).normal(0, 1, 100)
    result = clustering_coefficient(y, A, node)
    assert isinstance(result, dict)
