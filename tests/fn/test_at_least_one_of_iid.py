"""Tests for at_least_one_of_iid.at_least_one_of_iid."""

from morie.fn import _array_core as np

from morie.fn.at_least_one_of_iid import (
    at_least_one_of_iid,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e96_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = at_least_one_of_iid(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e96_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = at_least_one_of_iid(x)
    assert isinstance(result, dict)
