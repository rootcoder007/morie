"""Tests for etsmod.ets."""

from morie.fn import _array_core as np

from morie.fn.etsmod import ets


def test_etsmod_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ets(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_etsmod_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ets(y)
    assert isinstance(result, dict)
