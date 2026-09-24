"""Tests for tsalls.tsallis_entropy."""

from morie.fn import _array_core as np

from morie.fn.tsalls import tsallis_entropy


def test_tsalls_basic():
    """Test basic functionality."""
    p = np.random.default_rng(42).normal(0.0, 1.0, 40)
    q = 0.1
    result = tsallis_entropy(p, q)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_tsalls_edge():
    """Test edge cases."""
    p = np.random.default_rng(42).normal(0.0, 1.0, 40)
    q = 0.1
    result = tsallis_entropy(p, q)
    assert isinstance(result, dict)
