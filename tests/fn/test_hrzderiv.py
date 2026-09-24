"""Tests for hrzderiv.horowitz_density_derivative."""

from morie.fn import _array_core as np

from morie.fn.hrzderiv import horowitz_density_derivative


def test_hrzderiv_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_density_derivative(x)
    assert isinstance(result, dict)
    assert "grid" in result


def test_hrzderiv_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_density_derivative(x)
    assert isinstance(result, dict)
