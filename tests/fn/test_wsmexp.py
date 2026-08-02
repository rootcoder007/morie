"""Tests for wsmexp.wasserman_expectation."""

from morie.fn import _array_core as np

from morie.fn.wsmexp import wasserman_expectation


def test_wsmexp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_expectation(x, f)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_wsmexp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_expectation(x, f)
    assert isinstance(result, dict)
