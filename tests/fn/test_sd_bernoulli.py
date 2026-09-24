"""Tests for sd_bernoulli.sd_bernoulli."""

from morie.fn import _array_core as np

from morie.fn.sd_bernoulli import (
    sd_bernoulli,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e46_basic():
    """Test basic functionality."""
    p = 0.1
    result = sd_bernoulli(p)
    assert isinstance(result, dict)
    assert "p" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e46_edge():
    """Test edge cases."""
    p = 0.1
    result = sd_bernoulli(p)
    assert isinstance(result, dict)
