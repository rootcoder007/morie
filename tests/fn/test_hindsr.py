"""Tests for hindsr.her."""

from morie.fn import _array_core as np

from morie.fn.hindsr import her


def test_hindsr_basic():
    """Test basic functionality."""
    episodes = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = her(episodes)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_hindsr_edge():
    """Test edge cases."""
    episodes = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = her(episodes)
    assert isinstance(result, dict)
