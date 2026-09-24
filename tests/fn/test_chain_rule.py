"""Tests for chain_rule.chain_rule."""

import math

from morie.fn import _array_core as np
from morie.fn.chain_rule import chain_rule


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e9_basic():
    """Test basic functionality."""
    # Build a consistent set of probabilities so both factorizations agree.
    p_a = 0.5
    p_b_given_a = 0.4
    p_and = p_a * p_b_given_a  # = 0.2
    p_b = 0.25
    p_a_given_b = p_and / p_b  # = 0.8

    result = chain_rule(p_a, p_b_given_a, p_b, p_a_given_b)

    assert isinstance(result, dict)
    assert "via_a" in result
    assert "via_b" in result
    assert "p_and" in result
    assert math.isfinite(result["via_a"])
    assert math.isfinite(result["via_b"])
    assert math.isfinite(result["p_and"])
    assert math.isclose(result["via_a"], result["via_b"], abs_tol=1e-9)
    assert 0.0 <= result["p_and"] <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e9_edge():
    """Test edge cases with a small, uniform, valid input."""
    # All probabilities equal to 1/2 -> P(A and B) = 0.25 under both forms.
    result = chain_rule(0.5, 0.5, 0.5, 0.5)

    assert isinstance(result, dict)
    assert "via_a" in result
    assert "via_b" in result
    assert "p_and" in result
    assert math.isfinite(result["via_a"])
    assert math.isfinite(result["via_b"])
    assert 0.0 <= result["via_a"] <= 1.0
    assert math.isclose(result["via_a"], 0.25, abs_tol=1e-9)
