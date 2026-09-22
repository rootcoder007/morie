"""Tests for evgpdpw.evt_gpd_pwm."""

from morie.fn import _array_core as np

from morie.fn.evgpdpw import evt_gpd_pwm


def test_evgpdpw_basic():
    """Test basic functionality."""
    x = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = evt_gpd_pwm(x)
    assert isinstance(result, dict)
    assert "sigma" in result
def test_evgpdpw_edge():
    """Test edge cases."""
    x = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = evt_gpd_pwm(x)
    assert isinstance(result, dict)
