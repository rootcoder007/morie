"""Tests for sgtnlap.sgt_normalised_laplacian."""

from morie.fn import _array_core as np

from morie.fn.sgtnlap import sgt_normalised_laplacian


def test_sgtnlap_basic():
    """Test basic functionality."""
    W = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_normalised_laplacian(W)
    assert isinstance(result, dict)
    assert "Lcal" in result


def test_sgtnlap_edge():
    """Test edge cases."""
    W = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_normalised_laplacian(W)
    assert isinstance(result, dict)
