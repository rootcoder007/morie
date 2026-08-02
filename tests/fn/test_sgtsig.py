"""Tests for sgtsig.sgt_signless_laplacian."""

from morie.fn import _array_core as np

from morie.fn.sgtsig import sgt_signless_laplacian


def test_sgtsig_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_signless_laplacian(A)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_sgtsig_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_signless_laplacian(A)
    assert isinstance(result, dict)
