"""Tests for plncF.planck_function."""

from morie.fn import _array_core as np

from morie.fn.plncF import planck_function


def test_plncF_basic():
    """Test basic functionality."""
    lam = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    T = 0.1
    result = planck_function(lam, T)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_plncF_edge():
    """Test edge cases."""
    lam = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    T = 0.1
    result = planck_function(lam, T)
    assert isinstance(result, dict)
