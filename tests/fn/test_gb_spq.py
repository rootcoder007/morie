"""Tests for gb_spq.gibbons_spearman_test."""

from morie.fn import _array_core as np

from morie.fn.gb_spq import gibbons_spearman_test


def test_gb_spq_basic():
    """Test basic functionality."""
    r = 1.0
    n = 100
    result = gibbons_spearman_test(r, n)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_gb_spq_edge():
    """Test edge cases."""
    r = 1.0
    n = 100
    result = gibbons_spearman_test(r, n)
    assert isinstance(result, dict)
