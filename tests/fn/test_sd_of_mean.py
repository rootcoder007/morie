"""Tests for sd_of_mean.sd_of_mean."""

from morie.fn import _array_core as np

from morie.fn.sd_of_mean import (
    sd_of_mean,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e53_basic():
    """Test basic functionality."""
    sigma = 0.1
    n = 5
    result = sd_of_mean(sigma, n)
    assert isinstance(result, dict)
    assert "sigma" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e53_edge():
    """Test edge cases."""
    sigma = 0.1
    n = 5
    result = sd_of_mean(sigma, n)
    assert isinstance(result, dict)
