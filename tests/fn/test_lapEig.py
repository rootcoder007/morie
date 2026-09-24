"""Tests for lapEig.laplacian_eigenmaps."""

from morie.fn import _array_core as np

from morie.fn.lapEig import laplacian_eigenmaps


def test_lapEig_basic():
    """Test basic functionality."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = laplacian_eigenmaps(A)
    assert isinstance(result, dict)
    assert "embedding" in result


def test_lapEig_edge():
    """Test edge cases."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = laplacian_eigenmaps(A)
    assert isinstance(result, dict)
