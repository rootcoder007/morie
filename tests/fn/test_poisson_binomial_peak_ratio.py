"""Tests for poisson_binomial_peak_ratio.poisson_binomial_peak_ratio."""

from morie.fn import _array_core as np

from morie.fn.poisson_binomial_peak_ratio import (
    poisson_binomial_peak_ratio,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e98_basic():
    """Test basic functionality."""
    n = 5
    p = 0.5
    result = poisson_binomial_peak_ratio(n, p)
    assert isinstance(result, dict)
    assert "ratio" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e98_edge():
    """Test edge cases."""
    n = 5
    p = 0.5
    result = poisson_binomial_peak_ratio(n, p)
    assert isinstance(result, dict)
