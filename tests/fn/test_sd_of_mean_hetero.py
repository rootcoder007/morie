"""Tests for sd_of_mean_hetero.sd_of_mean_hetero."""

from morie.fn import _array_core as np

from morie.fn.sd_of_mean_hetero import (
    sd_of_mean_hetero,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e55_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = sd_of_mean_hetero(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e55_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = sd_of_mean_hetero(x)
    assert isinstance(result, dict)
