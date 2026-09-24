"""Tests for laplmo.laplacian_eigen."""

from morie.fn import _array_core as np

from morie.fn.laplmo import laplacian_eigen


def test_laplmo_basic():
    """Test basic functionality."""
    W = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = laplacian_eigen(W)
    assert isinstance(result, dict)
    assert "values" in result


def test_laplmo_edge():
    """Test edge cases."""
    W = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = laplacian_eigen(W)
    assert isinstance(result, dict)
