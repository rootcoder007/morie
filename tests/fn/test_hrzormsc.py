"""Tests for hrzormsc.horowitz_ordered_max_score."""

from morie.fn import _array_core as np

from morie.fn.hrzormsc import horowitz_ordered_max_score


def test_hrzormsc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = 5
    result = horowitz_ordered_max_score(x, y)
    assert isinstance(result, dict)
    assert "beta" in result


def test_hrzormsc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = 5
    result = horowitz_ordered_max_score(x, y)
    assert isinstance(result, dict)
