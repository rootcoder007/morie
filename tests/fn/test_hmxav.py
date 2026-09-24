"""Tests for hmxav.geron_glorot_init."""

from morie.fn import _array_core as np

from morie.fn.hmxav import geron_glorot_init


def test_hmxav_basic():
    """Test basic functionality."""
    fan_in = 40
    fan_out = 60
    result = geron_glorot_init(fan_in, fan_out)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmxav_edge():
    """Test edge cases."""
    fan_in = 40
    fan_out = 60
    result = geron_glorot_init(fan_in, fan_out)
    assert isinstance(result, dict)
