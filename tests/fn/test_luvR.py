"""Tests for luvR.louvain."""

from morie.fn import _array_core as np

from morie.fn.luvR import louvain


def test_luvR_basic():
    """Test basic functionality."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = louvain(A)
    assert isinstance(result, dict)
    assert "estimate" in result or "communities" in result


def test_luvR_edge():
    """Test edge cases."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = louvain(A)
    assert isinstance(result, dict)
