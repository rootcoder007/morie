"""Tests for fixed_effect_weight.fixed_effect_weight."""

from morie.fn import _array_core as np

from morie.fn.fixed_effect_weight import fixed_effect_weight


def test_ca11e34_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = fixed_effect_weight(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca11e34_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = fixed_effect_weight(x)
    assert isinstance(result, dict)
