"""Tests for sbmdg2.degree_corrected_sbm."""

from morie.fn import _array_core as np

from morie.fn.sbmdg2 import degree_corrected_sbm


def test_sbmdg2_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0.0, 1.0, 40)
    blocks = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = degree_corrected_sbm(A, blocks)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_sbmdg2_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0.0, 1.0, 40)
    blocks = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = degree_corrected_sbm(A, blocks)
    assert isinstance(result, dict)
