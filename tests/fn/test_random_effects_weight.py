"""Tests for random_effects_weight.random_effects_weight."""

from morie.fn import _array_core as np

from morie.fn.random_effects_weight import random_effects_weight


def test_ca11e43_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = random_effects_weight(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca11e43_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = random_effects_weight(x)
    assert isinstance(result, dict)
