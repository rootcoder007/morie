"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e28.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_28."""

import warnings

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e28 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_28,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e28_basic():
    """Test basic functionality."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_28()
    assert isinstance(result, dict)
    assert len(result) >= 1


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e28_edge():
    """Test edge cases."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_28()
    assert isinstance(result, dict)
    for value in result.values():
        assert value is not None
