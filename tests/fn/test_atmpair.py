"""Tests for atmpair.atom_pair_fp."""

from morie.fn import _array_core as np

from morie.fn.atmpair import atom_pair_fp


def test_atmpair_basic():
    """Test basic functionality."""
    adjacency = np.random.default_rng(42).normal(0, 1, (100, 100))
    atomtype = np.random.default_rng(42).normal(0, 1, 100)
    result = atom_pair_fp(adjacency, atomtype)
    assert isinstance(result, dict)
    assert "bits" in result
def test_atmpair_edge():
    """Test edge cases."""
    adjacency = np.random.default_rng(42).normal(0, 1, (100, 100))
    atomtype = np.random.default_rng(42).normal(0, 1, 100)
    result = atom_pair_fp(adjacency, atomtype)
    assert isinstance(result, dict)
