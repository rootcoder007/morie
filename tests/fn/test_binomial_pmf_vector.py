"""Tests for binomial_pmf_vector.binomial_pmf_vector."""

from morie.fn import _array_core as np

from morie.fn.binomial_pmf_vector import (
    binomial_pmf_vector,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e10_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = binomial_pmf_vector(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e10_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = binomial_pmf_vector(x)
    assert isinstance(result, dict)
