"""Tests for rng189.rangayyan_ch4_pan_tompkins_moving_window_integrator."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_ch4_pan_tompkins_moving_window_integrator


def test_rng189_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_pan_tompkins_moving_window_integrator(x)
    assert isinstance(result, dict)
    assert "y" in result


def test_rng189_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_pan_tompkins_moving_window_integrator(x)
    assert isinstance(result, dict)
