"""Tests for phpprn.phillips_perron."""

from morie.fn import _array_core as np

from morie.fn.phpprn import phillips_perron


def test_phpprn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = phillips_perron(y)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "statistic" in result


def test_phpprn_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = phillips_perron(y)
    assert isinstance(result, dict)
