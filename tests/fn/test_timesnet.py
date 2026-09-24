"""Tests for timesnet.timesnet."""

from morie.fn import _array_core as np

from morie.fn.timesnet import timesnet


def test_timesnet_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = timesnet(x)
    assert isinstance(result, dict)
    assert "frequency" in result


def test_timesnet_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = timesnet(x)
    assert isinstance(result, dict)
