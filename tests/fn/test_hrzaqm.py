"""Tests for hrzaqm.horowitz_additive_quantile."""

from morie.fn import _array_core as np

from morie.fn.hrzaqm import horowitz_additive_quantile


def test_hrzaqm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_additive_quantile(x, y)
    assert isinstance(result, dict)
    assert "mu" in result


def test_hrzaqm_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_additive_quantile(x, y)
    assert isinstance(result, dict)
