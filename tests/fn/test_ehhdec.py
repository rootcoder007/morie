"""Tests for ehhdec.ehh_decay."""

from morie.fn import _array_core as np

from morie.fn.ehhdec import ehh_decay


def test_ehhdec_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, L = 40, 20
    # Build a (N, L) 0/1 haplotype matrix (chromosomes by SNPs)
    haplotypes = rng.integers(0, 2, (n, L)).tolist()
    core = 10  # core SNP index (0-based)
    positions = np.linspace(0.0, 100.0, L)
    result = ehh_decay(haplotypes, core, positions)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "ehh1" in result
    assert "ehh0" in result
    assert "ehhs" in result
    assert "positions" in result
    assert "core" in result
    assert "n1" in result
    assert "n0" in result
    assert "n" in result
    assert "method" in result
    assert len(result["estimate"]) == L
    assert len(result["positions"]) == L
    assert result["n"] == n
    assert result["core"] == core
    assert result["n1"] + result["n0"] == n


def test_ehhdec_edge():
    """Test edge cases with default positions (positions=None)."""
    rng = np.random.default_rng(42)
    n, L = 40, 20
    haplotypes = rng.integers(0, 2, (n, L)).tolist()
    core = 5
    # Call without explicit positions; function should default to index positions
    result = ehh_decay(haplotypes, core)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "positions" in result
    assert len(result["estimate"]) == L
    assert len(result["positions"]) == L
    assert result["core"] == core
    assert result["n"] == n
