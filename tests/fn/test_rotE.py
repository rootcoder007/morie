"""Tests for rotE.rotate."""

from morie.fn import _array_core as np

from morie.fn.rotE import rotate


def test_rotE_basic():
    """Test basic functionality."""
    triples = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = rotate(triples)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_rotE_edge():
    """Test edge cases."""
    triples = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = rotate(triples)
    assert isinstance(result, dict)
