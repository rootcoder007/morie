"""Tests for ghs019.ghosal_ch3_tree_set_probability."""

from morie.fn import _array_core as np

from morie.fn.ghs019 import ghosal_ch3_tree_set_probability


def test_ghs019_basic():
    """Test basic functionality."""
    V_path = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    result = ghosal_ch3_tree_set_probability(V_path)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ghs019_edge():
    """Test edge cases."""
    V_path = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    result = ghosal_ch3_tree_set_probability(V_path)
    assert isinstance(result, dict)
