"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner2e66.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_66."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner2e66 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_66,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e66_basic():
    """Test basic functionality."""
    n = 50
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_66(n)
    assert isinstance(result, (int, float, dict))
    if isinstance(result, dict):
        value = next((v for v in result.values() if isinstance(v, (int, float))), None)
        assert value is not None
    else:
        value = result
    assert math.isfinite(value)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e66_edge():
    """Test edge cases."""
    n = 5
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_66(n)
    assert isinstance(result, (int, float, dict))
    if isinstance(result, dict):
        value = next((v for v in result.values() if isinstance(v, (int, float))), None)
        assert value is not None
    else:
        value = result
    assert math.isfinite(value)
