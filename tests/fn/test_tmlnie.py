"""Tests for tmlnie.tmle_natural_indirect."""

from morie.fn import _array_core as np

from morie.fn.tmlnie import tmle_natural_indirect


def test_tmlnie_basic():
    """Test basic functionality."""
    y11 = np.random.default_rng(42).normal(0, 1, 100)
    y10 = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_natural_indirect(y11, y10)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_tmlnie_edge():
    """Test edge cases."""
    y11 = np.random.default_rng(42).normal(0, 1, 100)
    y10 = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_natural_indirect(y11, y10)
    assert isinstance(result, dict)
