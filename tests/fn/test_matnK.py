"""Tests for matnK.matern_kernel."""

from morie.fn import _array_core as np

from morie.fn.matnK import matern_kernel


def test_matnK_basic():
    """Test basic functionality."""
    d = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    nu = 0.1
    rho = 0.1
    result = matern_kernel(d, nu, rho)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_matnK_edge():
    """Test edge cases."""
    d = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    nu = 0.1
    rho = 0.1
    result = matern_kernel(d, nu, rho)
    assert isinstance(result, dict)
