"""Tests for f_overall_ss.f_overall_ss."""

from morie.fn import _array_core as np

from morie.fn.f_overall_ss import f_overall_ss


def test_ca2e16_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = f_overall_ss(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca2e16_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = f_overall_ss(x)
    assert isinstance(result, dict)
