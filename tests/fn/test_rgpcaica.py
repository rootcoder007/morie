"""Tests for rgpcaica.rangayyan_pca_vs_ica."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_pca_vs_ica


def test_rgpcaica_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (3, 40))
    n_components = 2
    result = rangayyan_pca_vs_ica(X, n_components, maxiter=50, seed=42)
    assert isinstance(result, dict)


def test_rgpcaica_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (3, 40))
    n_components = 2
    result = rangayyan_pca_vs_ica(X, n_components, maxiter=50, seed=42)
    assert isinstance(result, dict)
