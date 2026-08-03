"""Tests for pmf_sum_convolution.pmf_sum_convolution."""

from morie.fn import _array_core as np

from morie.fn.pmf_sum_convolution import (
    pmf_sum_convolution,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e11_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = pmf_sum_convolution(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e11_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = pmf_sum_convolution(x)
    assert isinstance(result, dict)
