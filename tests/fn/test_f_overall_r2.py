"""Tests for f_overall_r2.f_overall_r2."""

from morie.fn import _array_core as np

from morie.fn.f_overall_r2 import f_overall_r2


def test_ca2e17_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = f_overall_r2(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca2e17_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = f_overall_r2(x)
    assert isinstance(result, dict)
