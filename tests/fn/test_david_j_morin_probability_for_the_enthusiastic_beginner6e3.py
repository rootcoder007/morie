"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner6e3.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_3."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e3 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_3,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e3_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    m = 0.5
    sigma_x = 1.0
    sigma_z = 0.5
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_3(m, sigma_x, sigma_z)
    assert isinstance(result, dict)
    assert len(result) > 0
    for value in result.values():
        if isinstance(value, (int, float)):
            assert math.isfinite(value)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e3_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    m = 0.0
    sigma_x = 0.5
    sigma_z = 0.5
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_3(m, sigma_x, sigma_z)
    assert isinstance(result, dict)
    assert len(result) > 0
    for value in result.values():
        if isinstance(value, (int, float)):
            assert math.isfinite(value)
