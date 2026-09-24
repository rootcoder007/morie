"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner7e21.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_21."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner7e21 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_21,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner7e21_basic():
    """Test basic functionality."""
    a = 0.5
    n = 10
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_21(a, n)
    assert isinstance(result, dict)
    assert "exact" in result
    assert math.isfinite(float(result["exact"]))


def test_david_j_morin_probability_for_the_enthusiastic_beginner7e21_edge():
    """Test edge cases."""
    a = 0.3
    n = 5
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_21(a, n, terms=20)
    assert isinstance(result, dict)
    assert "product_form" in result
    assert math.isfinite(float(result["product_form"]))
