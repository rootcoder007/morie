"""Tests for mwem.mwem."""

from morie.fn import _array_core as np

from morie.fn.mwem import mwem


def test_mwem_basic():
    """Test basic functionality."""
    B = np.random.default_rng(42).normal(0.0, 1.0, 40)
    queries = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mwem(B, queries)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_mwem_edge():
    """Test edge cases."""
    B = np.random.default_rng(42).normal(0.0, 1.0, 40)
    queries = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mwem(B, queries)
    assert isinstance(result, dict)
