"""Tests for conditional_from_joint.conditional_from_joint."""

import math

from morie.fn import _array_core as np

from morie.fn.conditional_from_joint import (
    conditional_from_joint,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e48_basic():
    """Test basic functionality."""
    p_a = 0.5
    p_a_and_b = 0.2
    result = conditional_from_joint(p_a_and_b, p_a)
    assert isinstance(result, dict)
    assert "p_a_and_b" in result
    assert "p_a" in result
    assert "p_b_given_a" in result
    assert math.isfinite(result["p_b_given_a"])
    assert 0.0 <= result["p_b_given_a"] <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e48_edge():
    """Test edge cases."""
    # Equal joint and marginal: P(B|A) should be 1.
    p_a = 0.3
    p_a_and_b = 0.3
    result = conditional_from_joint(p_a_and_b, p_a)
    assert isinstance(result, dict)
    assert "p_b_given_a" in result
    assert math.isclose(result["p_b_given_a"], 1.0)
    # Small but valid probability.
    p_a2 = 0.99
    p_a_and_b2 = 0.01
    result2 = conditional_from_joint(p_a_and_b2, p_a2)
    assert isinstance(result2, dict)
    assert math.isfinite(result2["p_b_given_a"])
    assert 0.0 <= result2["p_b_given_a"] <= 1.0
