"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner2e61.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_61."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner2e61 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_61,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e61_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_61(
        p_a=0.02, p_z_given_a=0.95, p_z_given_not_a=0.1
    )
    posterior = float(result["posterior"])
    assert math.isfinite(posterior)
    assert 0.0 <= posterior <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e61_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_61(
        p_a=0.5, p_z_given_a=0.8, p_z_given_not_a=0.2
    )
    posterior = float(result["posterior"])
    assert math.isfinite(posterior)
    assert 0.0 <= posterior <= 1.0
