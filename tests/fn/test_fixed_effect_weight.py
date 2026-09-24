"""Tests for fixed_effect_weight.fixed_effect_weight."""

import math

from morie.fn.fixed_effect_weight import fixed_effect_weight


def test_ca11e34_basic():
    """Test basic functionality."""
    se = 0.5
    result = fixed_effect_weight(se)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isclose(result["value"], 1.0 / (se ** 2))


def test_ca11e34_edge():
    """Test edge cases with a different se."""
    se = 2.0
    result = fixed_effect_weight(se)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] > 0
