"""Tests for gh_c5_4.ghosal_splitmerge."""

from morie.fn import _array_core as np

from morie.fn.gh_c5_4 import ghosal_splitmerge


def test_gh_c5_4_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    z_current = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_splitmerge(data, z_current)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_gh_c5_4_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    z_current = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_splitmerge(data, z_current)
    assert isinstance(result, dict)
