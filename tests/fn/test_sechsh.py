"""Tests for sechsh.hash_chain_audit."""

from morie.fn import _array_core as np

from morie.fn.sechsh import hash_chain_audit


def test_sechsh_basic():
    """Test basic functionality."""
    entries = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    hashes = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = hash_chain_audit(entries, hashes)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_sechsh_edge():
    """Test edge cases."""
    entries = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    hashes = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = hash_chain_audit(entries, hashes)
    assert isinstance(result, dict)
