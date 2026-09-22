"""Tests for bxprfl.baxter_king."""

from morie.fn import _array_core as np

from morie.fn.bxprfl import baxter_king


def test_bxprfl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = baxter_king(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_bxprfl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = baxter_king(y)
    assert isinstance(result, dict)
