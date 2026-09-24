"""Tests for resaln.resultant."""

from morie.fn import _array_core as np

from morie.fn.resaln import resultant


def test_resaln_basic():
    """Test basic functionality."""
    p = np.random.default_rng(42).normal(0.0, 1.0, 40)
    q = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = resultant(p, q)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_resaln_edge():
    """Test edge cases."""
    p = np.random.default_rng(42).normal(0.0, 1.0, 40)
    q = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = resultant(p, q)
    assert isinstance(result, dict)
