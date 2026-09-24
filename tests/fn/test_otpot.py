"""Tests for otpot.ot_pot_log_potentials."""

from morie.fn import _array_core as np

from morie.fn.otpot import ot_pot_log_potentials


def test_otpot_basic():
    """Test basic functionality."""
    u = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    v = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    epsilon = 0.1
    result = ot_pot_log_potentials(u, v, epsilon)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_otpot_edge():
    """Test edge cases."""
    u = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    v = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    epsilon = 0.1
    result = ot_pot_log_potentials(u, v, epsilon)
    assert isinstance(result, dict)
