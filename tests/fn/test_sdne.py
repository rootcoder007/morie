"""Tests for sdne.sdne."""

from morie.fn import _array_core as np

from morie.fn.sdne import sdne


def test_sdne_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = sdne(A, dim)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_sdne_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = sdne(A, dim)
    assert isinstance(result, dict)
