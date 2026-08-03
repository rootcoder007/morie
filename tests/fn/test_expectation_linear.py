"""Tests for expectation_linear.expectation_linear."""

from morie.fn import _array_core as np

from morie.fn.expectation_linear import (
    expectation_linear,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e13_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = expectation_linear(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e13_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = expectation_linear(x)
    assert isinstance(result, dict)
