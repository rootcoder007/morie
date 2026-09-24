"""Tests for hrzctrl.horowitz_control_function."""

from morie.fn import _array_core as np

from morie.fn.hrzctrl import horowitz_control_function


def test_hrzctrl_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    w = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_control_function(x, y, w)
    assert isinstance(result, dict)
    assert "g_hat" in result


def test_hrzctrl_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    w = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_control_function(x, y, w)
    assert isinstance(result, dict)
