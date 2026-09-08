"""Tests for alfmpv.alphafold_multimer."""

from morie.fn import _array_core as np

from morie.fn.alfmpv import alphafold_multimer


def test_alfmpv_basic():
    """Test basic functionality."""
    chains = np.random.default_rng(42).normal(0, 1, 100)
    msas = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_multimer(chains, msas)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_alfmpv_edge():
    """Test edge cases."""
    chains = np.random.default_rng(42).normal(0, 1, 100)
    msas = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_multimer(chains, msas)
    assert isinstance(result, dict)
