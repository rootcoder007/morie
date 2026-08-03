"""Tests for poisson_binomial_peak_ratio.poisson_binomial_peak_ratio."""

from morie.fn import _array_core as np

from morie.fn.poisson_binomial_peak_ratio import (
    poisson_binomial_peak_ratio,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e98_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = poisson_binomial_peak_ratio(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e98_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = poisson_binomial_peak_ratio(x)
    assert isinstance(result, dict)
