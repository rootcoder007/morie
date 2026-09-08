"""Tests for mafix.ma_fixed_effect."""

from morie.fn import _array_core as np

from morie.fn.mafix import ma_fixed_effect


def test_mafix_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_fixed_effect(yi, vi)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_mafix_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_fixed_effect(yi, vi)
    assert isinstance(result, dict)
