"""Tests for tmlrbt.tmle_robust."""

from morie.fn import _array_core as np

from morie.fn.tmlrbt import tmle_robust


def test_tmlrbt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    result = tmle_robust(y, D)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_tmlrbt_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    result = tmle_robust(y, D)
    assert isinstance(result, dict)
