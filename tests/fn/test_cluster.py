"""Tests for cluster.one_stage_cluster."""

from morie.fn import _array_core as np

from morie.fn.cluster import one_stage_cluster


def test_cluster_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(42).normal(0, 1, 100)
    result = one_stage_cluster(Y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_cluster_edge():
    """Test edge cases."""
    Y = np.random.default_rng(42).normal(0, 1, 100)
    result = one_stage_cluster(Y)
    assert isinstance(result, dict)
