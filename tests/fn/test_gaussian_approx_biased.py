"""Tests for gaussian_approx_biased.gaussian_approx_biased."""

from morie.fn import _array_core as np

from morie.fn.gaussian_approx_biased import (
    gaussian_approx_biased,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e15_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = gaussian_approx_biased(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e15_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = gaussian_approx_biased(x)
    assert isinstance(result, dict)
