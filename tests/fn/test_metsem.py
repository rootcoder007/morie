"""Tests for metsem.metagenome_assembly."""

from morie.fn import _array_core as np

from morie.fn.metsem import metagenome_assembly


def test_metsem_basic():
    """Test basic functionality."""
    reads = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    k = 5
    result = metagenome_assembly(reads, k)
    assert isinstance(result, dict)
    assert "contigs" in result


def test_metsem_edge():
    """Test edge cases."""
    reads = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    k = 5
    result = metagenome_assembly(reads, k)
    assert isinstance(result, dict)
