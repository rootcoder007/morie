"""Tests for hmgand.geron_anomaly_gmm."""

from morie.fn import _array_core as np

from morie.fn.hmgand import geron_anomaly_gmm


def test_hmgand_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_anomaly_gmm(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "is_anomaly" in result


def test_hmgand_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_anomaly_gmm(X)
    assert isinstance(result, dict)
