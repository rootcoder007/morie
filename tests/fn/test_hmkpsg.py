"""Tests for hmkpsg.geron_kernel_pca_sigmoid."""

from morie.fn import _array_core as np

from morie.fn.hmkpsg import geron_kernel_pca_sigmoid


def test_hmkpsg_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    n_components = 5
    result = geron_kernel_pca_sigmoid(X, n_components)
    assert isinstance(result, dict)
    assert "estimate" in result or "X_projected" in result


def test_hmkpsg_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    n_components = 5
    result = geron_kernel_pca_sigmoid(X, n_components)
    assert isinstance(result, dict)
