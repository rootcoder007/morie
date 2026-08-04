"""Tests for rgdtfd.rangayyan_decomp_tfd."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_decomp_tfd


def test_rgdtfd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    dictionary = np.random.default_rng(42).normal(0, 1, 100)
    max_atoms = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_decomp_tfd(x, fs, dictionary, max_atoms)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rgdtfd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    dictionary = np.random.default_rng(42).normal(0, 1, 100)
    max_atoms = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_decomp_tfd(x, fs, dictionary, max_atoms)
    assert isinstance(result, dict)
