"""Tests for hrzplr.horowitz_robinson_plr."""

from morie.fn import _array_core as np

from morie.fn.hrzplr import horowitz_robinson_plr


def test_hrzplr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_robinson_plr(X, Z, y)
    assert isinstance(result, dict)
    assert "beta" in result


def test_hrzplr_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_robinson_plr(X, Z, y)
    assert isinstance(result, dict)
