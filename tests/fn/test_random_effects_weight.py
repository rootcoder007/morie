"""Tests for random_effects_weight.random_effects_weight."""

from morie.fn import _array_core as np

from morie.fn.random_effects_weight import random_effects_weight


def test_ca11e43_basic():
    """Test basic functionality."""
    se = 0.5
    tau2 = 0.5
    result = random_effects_weight(se, tau2)
    assert isinstance(result, dict)
    assert "value" in result


def test_ca11e43_edge():
    """Test edge cases."""
    se = 0.5
    tau2 = 0.5
    result = random_effects_weight(se, tau2)
    assert isinstance(result, dict)
