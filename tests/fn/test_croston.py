"""Tests for croston.croston."""

from morie.fn import _array_core as np

from morie.fn.croston import croston


def test_croston_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = croston(y, alpha)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_croston_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = croston(y, alpha)
    assert isinstance(result, dict)
