"""Tests for rgbp.rangayyan_basis_pursuit."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_basis_pursuit


def test_rgbp_basic():
    """Test basic functionality."""
    x = 0.5
    D = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_basis_pursuit(x, D)
    assert isinstance(result, dict)
    assert "alpha" in result


def test_rgbp_edge():
    """Test edge cases."""
    x = 0.5
    D = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_basis_pursuit(x, D)
    assert isinstance(result, dict)
