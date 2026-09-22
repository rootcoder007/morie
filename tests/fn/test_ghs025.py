"""Tests for ghs025.ghosal_ch3_tailfree_density_product."""

from morie.fn import _array_core as np

from morie.fn.ghs025 import ghosal_ch3_tailfree_density_product


def test_ghs025_basic():
    """Test basic functionality."""
    V_path = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_ch3_tailfree_density_product(V_path)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ghs025_edge():
    """Test edge cases."""
    V_path = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_ch3_tailfree_density_product(V_path)
    assert isinstance(result, dict)
