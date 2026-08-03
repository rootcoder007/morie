"""Tests for pmf_sd.pmf_sd."""

from morie.fn import _array_core as np

from morie.fn.pmf_sd import (
    pmf_sd,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e31_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = pmf_sd(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e31_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = pmf_sd(x)
    assert isinstance(result, dict)
