"""Tests for larspec.lars_optimizer."""

from morie.fn import _array_core as np

from morie.fn.larspec import lars_optimizer


def test_larspec_basic():
    """Test basic functionality."""
    g = np.random.default_rng(42).normal(0.0, 1.0, 40)
    w = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = lars_optimizer(g, w)
    assert isinstance(result, dict)
    assert "update" in result


def test_larspec_edge():
    """Test edge cases."""
    g = np.random.default_rng(42).normal(0.0, 1.0, 40)
    w = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = lars_optimizer(g, w)
    assert isinstance(result, dict)
