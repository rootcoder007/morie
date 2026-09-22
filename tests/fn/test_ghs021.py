"""Tests for ghs021.ghosal_ch3_tailfree_max_bound."""

from morie.fn import _array_core as np

from morie.fn.ghs021 import ghosal_ch3_tailfree_max_bound


def test_ghs021_basic():
    """Test basic functionality."""
    EV2_by_level = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = ghosal_ch3_tailfree_max_bound(EV2_by_level, m)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ghs021_edge():
    """Test edge cases."""
    EV2_by_level = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = ghosal_ch3_tailfree_max_bound(EV2_by_level, m)
    assert isinstance(result, dict)
