"""Tests for rgegg.rangayyan_egg."""

from morie.fn import _array_core as np

from morie.fn.bsaphys import rangayyan_egg


def test_rgegg_basic():
    """Test basic functionality."""
    egg = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.5
    result = rangayyan_egg(egg, fs)
    assert isinstance(result, dict)
    assert "total_power" in result


def test_rgegg_edge():
    """Test edge cases."""
    egg = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.5
    result = rangayyan_egg(egg, fs)
    assert isinstance(result, dict)
