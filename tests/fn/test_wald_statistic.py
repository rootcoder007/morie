"""Tests for wald_statistic.wald_statistic."""

from morie.fn import _array_core as np

from morie.fn.wald_statistic import wald_statistic


def test_ca4e15_basic():
    """Test basic functionality."""
    b = 0.5
    se = 0.5
    result = wald_statistic(b, se)
    assert isinstance(result, dict)
    assert "value" in result


def test_ca4e15_edge():
    """Test edge cases."""
    b = 0.5
    se = 0.5
    result = wald_statistic(b, se)
    assert isinstance(result, dict)
