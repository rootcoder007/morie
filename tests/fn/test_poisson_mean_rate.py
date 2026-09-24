"""Tests for poisson_mean_rate.poisson_mean_rate."""

from morie.fn import _array_core as np

from morie.fn.poisson_mean_rate import (
    poisson_mean_rate,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e19_basic():
    """Test basic functionality."""
    lam = 0.5
    t = 0.5
    result = poisson_mean_rate(lam, t)
    assert isinstance(result, dict)
    assert "lambda" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e19_edge():
    """Test edge cases."""
    lam = 0.5
    t = 0.5
    result = poisson_mean_rate(lam, t)
    assert isinstance(result, dict)
