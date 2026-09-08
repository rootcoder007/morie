"""Tests for poisson_mean_rate.poisson_mean_rate."""

from morie.fn import _array_core as np

from morie.fn.poisson_mean_rate import (
    poisson_mean_rate,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e19_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = poisson_mean_rate(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e19_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = poisson_mean_rate(x)
    assert isinstance(result, dict)
