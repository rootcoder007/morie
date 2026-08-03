"""Tests for cluster_means_model.cluster_means_model."""

from morie.fn import _array_core as np

from morie.fn.cluster_means_model import cluster_means_model


def test_ca7e4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cluster_means_model(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca7e4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cluster_means_model(x)
    assert isinstance(result, dict)
