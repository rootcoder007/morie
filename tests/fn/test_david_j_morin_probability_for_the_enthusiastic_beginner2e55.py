"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner2e55.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_55."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner2e55 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_55,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e55_basic():
    """Test basic functionality."""
    p_a = 0.3
    p_z_given_a = 0.8
    p_z_given_not_a = 0.2
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_55(
        p_a, p_z_given_a, p_z_given_not_a
    )
    assert isinstance(result, dict)
    assert len(result) >= 1
    for value in result.values():
        assert math.isfinite(float(value))
        assert 0.0 <= float(value) <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e55_edge():
    """Test edge cases."""
    # extreme probabilities
    p_a = 0.0
    p_z_given_a = 0.5
    p_z_given_not_a = 0.5
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_55(
        p_a, p_z_given_a, p_z_given_not_a
    )
    assert isinstance(result, dict)
    for value in result.values():
        assert math.isfinite(float(value))
        assert 0.0 <= float(value) <= 1.0
