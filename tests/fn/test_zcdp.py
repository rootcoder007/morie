"""Tests for zcdp.zcdp."""

from morie.fn import _array_core as np

from morie.fn.zcdp import zcdp


def test_zcdp_basic():
    """Test basic functionality."""
    mech = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    rho = 0.1
    result = zcdp(mech, rho)
    assert isinstance(result, dict)
    assert "rho_total" in result


def test_zcdp_edge():
    """Test edge cases."""
    mech = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    rho = 0.1
    result = zcdp(mech, rho)
    assert isinstance(result, dict)
