"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner2e29.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_29."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner2e29 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_29,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e29_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 5
    raw = rng.uniform(0.0, 1.0, n)
    priors = [r / np.sum(raw) for r in raw]
    likelihoods = rng.uniform(0.0, 1.0, n)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_29(
        priors, likelihoods
    )
    assert isinstance(result, dict)
    assert "p_total" in result
    assert math.isfinite(result["p_total"])
    assert 0.0 <= result["p_total"] <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e29_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 3
    raw = rng.uniform(0.0, 1.0, n)
    priors = [r / np.sum(raw) for r in raw]
    likelihoods = rng.uniform(0.0, 1.0, n)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_29(
        priors, likelihoods
    )
    assert isinstance(result, dict)
    assert "p_total" in result
    assert math.isfinite(result["p_total"])
    assert 0.0 <= result["p_total"] <= 1.0
