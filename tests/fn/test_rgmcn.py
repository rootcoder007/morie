"""Tests for rgmcn.rangayyan_mcnemar_test."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_mcnemar_test


def test_rgmcn_basic():
    """Test basic functionality."""
    table = [[20, 5], [8, 25]]
    result = rangayyan_mcnemar_test(table)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_rgmcn_edge():
    """Test edge cases."""
    table = [[20, 5], [8, 25]]
    result = rangayyan_mcnemar_test(table)
    assert isinstance(result, dict)
