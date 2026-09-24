"""Tests for sgtlap2.sgt_laplacian_eigenmaps."""

from morie.fn import _array_core as np

from morie.fn.sgtlap2 import sgt_laplacian_eigenmaps


def test_sgtlap2_basic():
    """Test basic functionality."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_laplacian_eigenmaps(A)
    assert isinstance(result, dict)
    assert "Y" in result


def test_sgtlap2_edge():
    """Test edge cases."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_laplacian_eigenmaps(A)
    assert isinstance(result, dict)
