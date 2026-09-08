"""Tests for catstop.cat_stopping_rule."""

from morie.fn import _array_core as np

from morie.fn.catstop import cat_stopping_rule


def test_catstop_basic():
    """Test basic functionality."""
    theta = 0.0
    items_taken = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = cat_stopping_rule(theta, items_taken, threshold)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_catstop_edge():
    """Test edge cases."""
    theta = 0.0
    items_taken = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = cat_stopping_rule(theta, items_taken, threshold)
    assert isinstance(result, dict)
