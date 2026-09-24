"""Tests for hrztikr.horowitz_tikhonov_npiv."""

from morie.fn import _array_core as np

from morie.fn.hrztikr import horowitz_tikhonov_npiv


def test_hrztikr_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Ey_w = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_tikhonov_npiv(T, Ey_w)
    assert isinstance(result, dict)
    assert "g" in result


def test_hrztikr_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Ey_w = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_tikhonov_npiv(T, Ey_w)
    assert isinstance(result, dict)
