"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e51.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_51."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e51 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_51,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e51_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_51(10)
    assert isinstance(result, dict)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e51_edge():
    """Test edge cases."""
    with pytest.raises(Exception):
        david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_51(-1)
