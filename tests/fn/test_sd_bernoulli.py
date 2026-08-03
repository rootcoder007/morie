"""Tests for sd_bernoulli.sd_bernoulli."""

from morie.fn import _array_core as np

from morie.fn.sd_bernoulli import (
    sd_bernoulli,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e46_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = sd_bernoulli(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e46_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = sd_bernoulli(x)
    assert isinstance(result, dict)
