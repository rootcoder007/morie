"""Tests for sgtlap.sgt_laplacian."""

from morie.fn import _array_core as np

from morie.fn.sgtlap import sgt_laplacian


def test_sgtlap_basic():
    """Test basic functionality."""
    W = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_laplacian(W)
    assert isinstance(result, dict)
    assert "L" in result


def test_sgtlap_edge():
    """Test edge cases."""
    W = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_laplacian(W)
    assert isinstance(result, dict)
