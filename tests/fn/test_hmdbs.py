"""Tests for hmdbs.geron_dbscan."""

from morie.fn import _array_core as np

from morie.fn.hmdbs import geron_dbscan


def test_hmdbs_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    eps = 0.1
    min_samples = 5
    result = geron_dbscan(X, eps, min_samples)
    assert isinstance(result, dict)
    assert "estimate" in result or "labels" in result


def test_hmdbs_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    eps = 0.1
    min_samples = 5
    result = geron_dbscan(X, eps, min_samples)
    assert isinstance(result, dict)
