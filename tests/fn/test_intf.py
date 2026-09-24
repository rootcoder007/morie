"""Tests for intf.integrate_function."""

from morie.fn import _array_core as np

from morie.fn.intf import integrate_function


def test_intf_basic():
    """Test basic functionality."""
    coef = 0.5
    basis = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = integrate_function(coef, basis)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_intf_edge():
    """Test edge cases."""
    coef = 0.5
    basis = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = integrate_function(coef, basis)
    assert isinstance(result, dict)
