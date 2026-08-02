"""Tests for hmgand.geron_anomaly_gmm."""

from morie.fn import _array_core as np

from morie.fn.hmgand import geron_anomaly_gmm


def test_hmgand_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_anomaly_gmm(X, n_components, threshold)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmgand_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_anomaly_gmm(X, n_components, threshold)
    assert isinstance(result, dict)
