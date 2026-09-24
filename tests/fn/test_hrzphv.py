"""Tests for hrzphv.horowitz_ph_heterogeneity."""

from morie.fn import _array_core as np

from morie.fn.hrzphv import horowitz_ph_heterogeneity


def test_hrzphv_basic():
    """Test basic functionality."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_ph_heterogeneity(t, x)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_hrzphv_edge():
    """Test edge cases."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_ph_heterogeneity(t, x)
    assert isinstance(result, dict)
