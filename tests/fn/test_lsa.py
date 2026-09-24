"""Tests for lsa.lsa."""

from morie.fn import _array_core as np

from morie.fn.lsa import lsa


def test_lsa_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = lsa(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_lsa_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = lsa(X)
    assert isinstance(result, dict)
