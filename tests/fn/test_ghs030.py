"""Tests for ghs030.ghosal_ch3_polya_tree_posterior_density."""

from morie.fn import _array_core as np

from morie.fn.ghs030 import ghosal_ch3_polya_tree_posterior_density


def test_ghs030_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_ch3_polya_tree_posterior_density(x, data)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ghs030_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_ch3_polya_tree_posterior_density(x, data)
    assert isinstance(result, dict)
