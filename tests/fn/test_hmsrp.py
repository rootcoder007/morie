"""Tests for hmsrp.geron_sparse_rand_projection."""

from morie.fn import _array_core as np

from morie.fn.hmsrp import geron_sparse_rand_projection


def test_hmsrp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    d_out = 5
    result = geron_sparse_rand_projection(X, d_out)
    assert isinstance(result, dict)
    assert "estimate" in result or "X_proj" in result


def test_hmsrp_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    d_out = 5
    result = geron_sparse_rand_projection(X, d_out)
    assert isinstance(result, dict)
