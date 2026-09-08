"""Tests for conditional_from_joint.conditional_from_joint."""

from morie.fn import _array_core as np

from morie.fn.conditional_from_joint import (
    conditional_from_joint,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e48_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = conditional_from_joint(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e48_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = conditional_from_joint(x)
    assert isinstance(result, dict)
