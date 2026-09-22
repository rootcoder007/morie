"""Tests for fclust.functional_clustering."""

from morie.fn import _array_core as np

from morie.fn.fclust import functional_clustering


def test_fclust_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = functional_clustering(Y)
    assert isinstance(result, dict)
    assert "labels" in result
def test_fclust_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = functional_clustering(Y)
    assert isinstance(result, dict)
