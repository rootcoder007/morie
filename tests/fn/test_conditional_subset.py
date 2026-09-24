"""Tests for conditional_subset.conditional_subset."""

import math

from morie.fn import _array_core as np

from morie.fn.conditional_subset import (
    conditional_subset,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e49_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    p_a = rng.uniform(0.2, 0.9)
    p_b = rng.uniform(0.0, p_a)
    result = conditional_subset(p_b, p_a)
    assert isinstance(result, dict)
    # Verify the payload contains the expected keys
    assert "p_b" in result
    assert "p_a" in result
    assert "p_b_given_a" in result
    # The conditional probability should equal p_b / p_a
    expected = p_b / p_a
    assert math.isclose(result["p_b_given_a"], expected, rel_tol=1e-9, abs_tol=1e-12)
    # The result should be a finite probability in [0, 1]
    assert math.isfinite(result["p_b_given_a"])
    assert 0.0 <= result["p_b_given_a"] <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e49_edge():
    """Test edge cases."""
    # Edge case: p_b is zero → conditional probability is zero
    result_zero = conditional_subset(0.0, 0.7)
    assert isinstance(result_zero, dict)
    assert "p_b_given_a" in result_zero
    assert math.isclose(result_zero["p_b_given_a"], 0.0, abs_tol=1e-12)
    assert math.isfinite(result_zero["p_b_given_a"])

    # Edge case: p_b equals p_a → conditional probability is one
    p = 0.3
    result_one = conditional_subset(p, p)
    assert isinstance(result_one, dict)
    assert math.isclose(result_one["p_b_given_a"], 1.0, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isfinite(result_one["p_b_given_a"])
