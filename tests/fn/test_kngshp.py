"""Tests for kngshp.kinship_estimator."""

from morie.fn import _array_core as np

from morie.fn.kngshp import kinship_estimator


def test_kngshp_basic():
    """Test basic functionality."""
    genotypes = np.random.default_rng(42).normal(0, 1, 100)
    result = kinship_estimator(genotypes)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kngshp_edge():
    """Test edge cases."""
    genotypes = np.random.default_rng(42).normal(0, 1, 100)
    result = kinship_estimator(genotypes)
    assert isinstance(result, dict)
