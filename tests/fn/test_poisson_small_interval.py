"""Tests for poisson_small_interval.poisson_small_interval."""

from morie.fn import _array_core as np

from morie.fn.poisson_small_interval import (
    poisson_small_interval,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e18_basic():
    """Test basic functionality."""
    lam = 0.5
    eps = 0.5
    result = poisson_small_interval(lam, eps)
    assert isinstance(result, dict)
    assert "approx" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e18_edge():
    """Test edge cases."""
    lam = 0.5
    eps = 0.5
    result = poisson_small_interval(lam, eps)
    assert isinstance(result, dict)
