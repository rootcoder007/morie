"""Tests for sgtkpc.sgt_kernel_pca."""

from morie.fn import _array_core as np

from morie.fn.sgtkpc import sgt_kernel_pca


def test_sgtkpc_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = sgt_kernel_pca(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "Y" in result


def test_sgtkpc_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = sgt_kernel_pca(X)
    assert isinstance(result, dict)
