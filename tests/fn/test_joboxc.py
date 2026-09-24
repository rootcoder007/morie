"""Tests for joboxc.joseph_box_cox_transform."""

from morie.fn import _array_core as np

from morie.fn.joboxc import joseph_box_cox_transform


def test_joboxc_basic():
    """Test basic functionality."""
    x = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    lam = 0.5
    result = joseph_box_cox_transform(x, lam)
    assert isinstance(result, dict)
    assert "w" in result


def test_joboxc_edge():
    """Test edge cases."""
    x = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    lam = 0.5
    result = joseph_box_cox_transform(x, lam)
    assert isinstance(result, dict)
