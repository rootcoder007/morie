"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner2e86.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_86."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner2e86 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_86,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e86_basic():
    """Test basic functionality."""
    p_a = 0.4
    p_b_given_a = 0.7
    p_b_given_not_a = 0.2
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_86(
            p_a, p_b_given_a, p_b_given_not_a
        )
    p_total = result["p_total"]
    assert math.isfinite(p_total)
    assert 0.0 <= p_total <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e86_edge():
    """Test edge cases."""
    p_a = 0.0
    p_b_given_a = 0.5
    p_b_given_not_a = 0.5
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_86(
            p_a, p_b_given_a, p_b_given_not_a
        )
    p_total = result["p_total"]
    assert math.isfinite(p_total)
    assert 0.0 <= p_total <= 1.0
