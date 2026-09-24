"""Tests for polyCh.chebyshev_basis."""

from morie.fn import _array_core as np

from morie.fn.polyCh import chebyshev_basis


def test_polyCh_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = chebyshev_basis(x)
    assert isinstance(result, dict)
    assert "basis" in result


def test_polyCh_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = chebyshev_basis(x)
    assert isinstance(result, dict)
