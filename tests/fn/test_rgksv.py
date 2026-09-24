"""Tests for rgksv.rangayyan_ksvd."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_ksvd


def test_rgksv_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    natoms = 5
    sparsity = 5
    result = rangayyan_ksvd(Y, natoms, sparsity)
    assert isinstance(result, dict)
    assert "dictionary" in result


def test_rgksv_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    natoms = 5
    sparsity = 5
    result = rangayyan_ksvd(Y, natoms, sparsity)
    assert isinstance(result, dict)
