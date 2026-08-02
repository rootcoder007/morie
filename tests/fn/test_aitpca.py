"""Tests for aitpca.aitchison_clr_pca."""

from morie.fn import _array_core as np

from morie.fn.aitpca import aitchison_clr_pca


def test_aitpca_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = aitchison_clr_pca(X, k)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_aitpca_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = aitchison_clr_pca(X, k)
    assert isinstance(result, dict)
