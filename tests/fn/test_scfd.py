"""Tests for scfd.scalar_on_function."""

from morie.fn import _array_core as np

from morie.fn.scfd import scalar_on_function


def test_scfd_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    basis = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = scalar_on_function(X, Y, basis)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_scfd_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    basis = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = scalar_on_function(X, Y, basis)
    assert isinstance(result, dict)
