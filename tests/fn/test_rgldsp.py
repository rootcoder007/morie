"""Tests for rgldsp.rangayyan_dictionary_sparse."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_dictionary_sparse


def test_rgldsp_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    D = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    sparsity = 5
    result = rangayyan_dictionary_sparse(Y, D, sparsity)
    assert isinstance(result, dict)
    assert "coefficients" in result


def test_rgldsp_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    D = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    sparsity = 5
    result = rangayyan_dictionary_sparse(Y, D, sparsity)
    assert isinstance(result, dict)
