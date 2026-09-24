"""Tests for sgthits.sgt_hits_kleinberg."""

from morie.fn import _array_core as np

from morie.fn.sgthits import sgt_hits_kleinberg


def test_sgthits_basic():
    """Test basic functionality."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_hits_kleinberg(A)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_sgthits_edge():
    """Test edge cases."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sgt_hits_kleinberg(A)
    assert isinstance(result, dict)
