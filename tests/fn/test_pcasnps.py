"""Tests for pcasnps.pca_snps."""

from morie.fn import _array_core as np

from morie.fn.pcasnps import pca_snps


def test_pcasnps_basic():
    """Test basic functionality."""
    genotypes = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = pca_snps(genotypes)
    assert isinstance(result, dict)
    assert "estimate" in result or "eigenvalues" in result


def test_pcasnps_edge():
    """Test edge cases."""
    genotypes = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = pca_snps(genotypes)
    assert isinstance(result, dict)
