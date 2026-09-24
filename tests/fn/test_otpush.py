"""Tests for otpush.ot_pushforward_density."""

from morie.fn import _array_core as np

from morie.fn.otpush import ot_pushforward_density


def test_otpush_basic():
    """Test basic functionality."""
    mu_grid = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    T_jac = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    T_inv_grid = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ot_pushforward_density(mu_grid, T_jac, T_inv_grid)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_otpush_edge():
    """Test edge cases."""
    mu_grid = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    T_jac = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    T_inv_grid = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ot_pushforward_density(mu_grid, T_jac, T_inv_grid)
    assert isinstance(result, dict)
