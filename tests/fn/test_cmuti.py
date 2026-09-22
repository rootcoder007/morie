"""Tests for cmuti.copula_mutual_information."""

from morie.fn import _array_core as np

from morie.fn.cmuti import copula_mutual_information


def test_cmuti_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = copula_mutual_information(x, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_cmuti_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = copula_mutual_information(x, y)
    assert isinstance(result, dict)
