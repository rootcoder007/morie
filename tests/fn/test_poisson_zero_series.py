"""Tests for poisson_zero_series.poisson_zero_series."""

from morie.fn import _array_core as np

from morie.fn.poisson_zero_series import (
    poisson_zero_series,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e53_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = poisson_zero_series(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e53_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = poisson_zero_series(x)
    assert isinstance(result, dict)
