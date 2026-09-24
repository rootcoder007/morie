"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner2e58.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_58."""

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner2e58 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_58,
)

import math


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e58_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_58(
        p_a=0.02, p_z_given_a=0.95, p_z_given_not_a=0.1
    )
    assert isinstance(result, dict)
    assert "posterior" in result
    posterior = result["posterior"]
    assert math.isfinite(posterior)
    assert 0.0 <= posterior <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e58_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_58(
        p_a=0.5, p_z_given_a=1.0, p_z_given_not_a=0.0
    )
    assert isinstance(result, dict)
    assert "posterior" in result
    posterior = result["posterior"]
    assert math.isfinite(posterior)
    assert 0.0 <= posterior <= 1.0
