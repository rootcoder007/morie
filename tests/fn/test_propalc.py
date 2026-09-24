"""Tests for propalc.proportional_allocation."""

from morie.fn import _array_core as np

from morie.fn.propalc import proportional_allocation


def test_propalc_basic():
    """Test basic functionality."""
    Nh = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    n = 5
    result = proportional_allocation(Nh, n)
    assert isinstance(result, dict)
    assert "nh" in result


def test_propalc_edge():
    """Test edge cases."""
    Nh = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    n = 5
    result = proportional_allocation(Nh, n)
    assert isinstance(result, dict)
