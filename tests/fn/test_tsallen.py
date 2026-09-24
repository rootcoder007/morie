"""Tests for tsallen.tsallis_entropy."""

from morie.fn import _array_core as np

from morie.fn.tsallen import tsallis_entropy


def test_tsallen_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    q = 0.1
    result = tsallis_entropy(y, q)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_tsallen_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    q = 0.1
    result = tsallis_entropy(y, q)
    assert isinstance(result, dict)
