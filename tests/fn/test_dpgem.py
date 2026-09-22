"""Tests for dpgem.gem_distribution."""

from morie.fn import _array_core as np

from morie.fn.dpgem import gem_distribution


def test_dpgem_basic():
    """Test basic functionality."""
    result = gem_distribution()
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_dpgem_edge():
    """Test edge cases."""
    result = gem_distribution()
    assert isinstance(result, dict)
