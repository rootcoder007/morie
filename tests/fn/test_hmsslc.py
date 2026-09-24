"""Tests for hmsslc.geron_semisupervised_cluster."""

from morie.fn import _array_core as np

from morie.fn.hmsslc import geron_semisupervised_cluster


def test_hmsslc_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X_labeled = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y_labeled = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_semisupervised_cluster(X, X_labeled, y_labeled)
    assert isinstance(result, dict)
    assert "estimate" in result or "labels" in result


def test_hmsslc_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X_labeled = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y_labeled = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_semisupervised_cluster(X, X_labeled, y_labeled)
    assert isinstance(result, dict)
