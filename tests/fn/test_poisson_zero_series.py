"""Tests for poisson_zero_series.poisson_zero_series."""

from morie.fn import _array_core as np

from morie.fn.poisson_zero_series import (
    poisson_zero_series,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e53_basic():
    """Test basic functionality."""
    a = 0.5
    result = poisson_zero_series(a)
    assert isinstance(result, dict)
    assert "partial_sums" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e53_edge():
    """Test edge cases."""
    a = 0.5
    result = poisson_zero_series(a)
    assert isinstance(result, dict)
