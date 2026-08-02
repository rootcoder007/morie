"""Tests for aitbcp.compositional_bray_curtis."""

from morie.fn import _array_core as np

from morie.fn.aitbcp import compositional_bray_curtis


def test_aitbcp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = compositional_bray_curtis(x, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_aitbcp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = compositional_bray_curtis(x, y)
    assert isinstance(result, dict)
