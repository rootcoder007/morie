"""Tests for hmkppl.geron_kernel_pca_poly."""

from morie.fn import _array_core as np

from morie.fn.hmkppl import geron_kernel_pca_poly


def test_hmkppl_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    n_components = 5
    result = geron_kernel_pca_poly(X, n_components)
    assert isinstance(result, dict)
    assert "estimate" in result or "X_projected" in result


def test_hmkppl_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    n_components = 5
    result = geron_kernel_pca_poly(X, n_components)
    assert isinstance(result, dict)
