"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e19.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_19."""

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e19 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_19,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e19_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    values = rng.integers(0, 10, 5)
    probs = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_19(values, probs)
    assert isinstance(result, dict)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e19_edge():
    """Test edge cases."""
    values = [0, 1]
    probs = np.array([0.5, 0.5])
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_19(values, probs)
    assert isinstance(result, dict)
