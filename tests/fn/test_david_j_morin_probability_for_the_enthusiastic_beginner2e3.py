"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner2e3.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_3."""

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner2e3 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_3,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e3_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    raw = rng.uniform(0, 1, 5)
    total = sum(raw)
    ps = [float(x) / total for x in raw]
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_3(ps)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e3_edge():
    """Test edge cases."""
    ps = [0.25, 0.25, 0.25, 0.25]
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_3(ps)
    assert isinstance(result, dict)
    assert len(result) > 0
