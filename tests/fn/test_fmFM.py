"""Tests for fmFM.factorization_machines."""

from morie.fn import _array_core as np

from morie.fn.fmFM import factorization_machines


def test_fmFM_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = factorization_machines(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_fmFM_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = factorization_machines(X, y)
    assert isinstance(result, dict)
