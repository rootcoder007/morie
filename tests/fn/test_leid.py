"""Tests for leid.leiden_communities."""

from morie.fn import _array_core as np

from morie.fn.leid import leiden_communities


def test_leid_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = leiden_communities(y, A)
    assert isinstance(result, dict)
    assert "estimate" in result or "labels" in result


def test_leid_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = leiden_communities(y, A)
    assert isinstance(result, dict)
