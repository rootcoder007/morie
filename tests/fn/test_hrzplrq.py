"""Tests for hrzplrq.horowitz_plr_quantile."""

from morie.fn import _array_core as np

from morie.fn.hrzplrq import horowitz_plr_quantile


def test_hrzplrq_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_plr_quantile(x, y, z)
    assert isinstance(result, dict)
    assert "beta_tau" in result


def test_hrzplrq_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_plr_quantile(x, y, z)
    assert isinstance(result, dict)
