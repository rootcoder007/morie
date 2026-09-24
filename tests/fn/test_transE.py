"""Tests for transE.transe."""

from morie.fn import _array_core as np

from morie.fn.transE import transe


def test_transE_basic():
    """Test basic functionality."""
    triples = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    dim = 5
    result = transe(triples, dim)
    assert isinstance(result, dict)
    assert "loss" in result


def test_transE_edge():
    """Test edge cases."""
    triples = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    dim = 5
    result = transe(triples, dim)
    assert isinstance(result, dict)
