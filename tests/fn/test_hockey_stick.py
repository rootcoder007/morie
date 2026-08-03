"""Tests for hockey_stick.hockey_stick."""

from morie.fn import _array_core as np

from morie.fn.hockey_stick import (
    hockey_stick,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e29_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hockey_stick(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e29_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hockey_stick(x)
    assert isinstance(result, dict)
