"""Tests for infmer.informer."""

from morie.fn import _array_core as np

from morie.fn.infmer import informer


def test_infmer_basic():
    """Test basic functionality."""
    Q = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    K = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    V = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = informer(Q, K, V)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_infmer_edge():
    """Test edge cases."""
    Q = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    K = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    V = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = informer(Q, K, V)
    assert isinstance(result, dict)
