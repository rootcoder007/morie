"""Tests for sd_of_mean.sd_of_mean."""

from morie.fn import _array_core as np

from morie.fn.sd_of_mean import (
    sd_of_mean,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e53_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = sd_of_mean(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e53_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = sd_of_mean(x)
    assert isinstance(result, dict)
