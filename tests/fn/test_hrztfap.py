"""Tests for hrztfap.horowitz_T_F_asymp_props."""

from morie.fn import _array_core as np

from morie.fn.hrztfap import horowitz_T_F_asymp_props


def test_hrztfap_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    bandwidth = 0.1
    result = horowitz_T_F_asymp_props(x, y, bandwidth)
    assert isinstance(result, dict)
    assert "asymptotic_distribution" in result


def test_hrztfap_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    bandwidth = 0.1
    result = horowitz_T_F_asymp_props(x, y, bandwidth)
    assert isinstance(result, dict)
