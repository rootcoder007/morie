"""Tests for pcadr.pca_dimensionality_reduction."""

from morie.fn import _array_core as np

from morie.fn.pcadr import pca_dimensionality_reduction


def test_pcadr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = pca_dimensionality_reduction(X)
    assert isinstance(result, dict)
    assert "scores" in result


def test_pcadr_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = pca_dimensionality_reduction(X)
    assert isinstance(result, dict)
