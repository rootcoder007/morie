"""Tests for hrzrateG.horowitz_rate_G_estimation."""

from morie.fn import _array_core as np

from morie.fn.hrzrateG import horowitz_rate_G_estimation


def test_hrzrateG_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    beta = 0.1
    result = horowitz_rate_G_estimation(x, y, beta)
    assert isinstance(result, dict)
    assert "grid" in result


def test_hrzrateG_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    beta = 0.1
    result = horowitz_rate_G_estimation(x, y, beta)
    assert isinstance(result, dict)
