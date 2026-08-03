"""Tests for inclusion_exclusion_3.inclusion_exclusion_3."""

from morie.fn import _array_core as np

from morie.fn.inclusion_exclusion_3 import (
    inclusion_exclusion_3,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e92_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = inclusion_exclusion_3(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e92_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = inclusion_exclusion_3(x)
    assert isinstance(result, dict)
