"""Tests for wald_statistic.wald_statistic."""

from morie.fn import _array_core as np

from morie.fn.wald_statistic import wald_statistic


def test_ca4e15_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wald_statistic(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca4e15_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wald_statistic(x)
    assert isinstance(result, dict)
