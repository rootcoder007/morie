"""Tests for hmipca.geron_incremental_pca."""

from morie.fn import _array_core as np

from morie.fn.hmipca import geron_incremental_pca


def test_hmipca_basic():
    """Test basic functionality."""
    X_iter = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    n_components = 5
    result = geron_incremental_pca(X_iter, n_components)
    assert isinstance(result, dict)
    assert "estimate" in result or "components" in result


def test_hmipca_edge():
    """Test edge cases."""
    X_iter = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    n_components = 5
    result = geron_incremental_pca(X_iter, n_components)
    assert isinstance(result, dict)
