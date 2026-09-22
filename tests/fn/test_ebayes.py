"""Tests for ebayes.empirical_bayes_shrinkage."""

from morie.fn import _array_core as np

from morie.fn.ebayes import empirical_bayes_shrinkage


def test_ebayes_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = empirical_bayes_shrinkage(y, cluster)
    assert isinstance(result, dict)
    assert "clusters" in result
def test_ebayes_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = empirical_bayes_shrinkage(y, cluster)
    assert isinstance(result, dict)
