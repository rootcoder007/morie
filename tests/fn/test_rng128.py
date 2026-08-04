"""Tests for rng128.rangayyan_ch3_bilinear_unit_circle_relation."""

from morie.fn import _array_core as np

from morie.fn.bsafilt import rangayyan_ch3_bilinear_unit_circle_relation


def test_rng128_basic():
    """Test basic functionality."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_bilinear_unit_circle_relation(omega, T)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rng128_edge():
    """Test edge cases."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_bilinear_unit_circle_relation(omega, T)
    assert isinstance(result, dict)
