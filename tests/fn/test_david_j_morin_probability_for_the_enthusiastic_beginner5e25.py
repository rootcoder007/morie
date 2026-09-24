"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner5e25.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_25."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner5e25 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_25,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e25_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_25(n_sigmas=3.0, sigma=1.0)
    assert isinstance(result, dict)
    assert "n_sigmas" in result
    assert "area_fraction" in result
    area = float(result["area_fraction"])
    assert math.isfinite(area)
    assert 0.0 <= area <= 1.0
    assert math.isfinite(float(result["n_sigmas"]))


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e25_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_25(n_sigmas=0.5, sigma=2.0)
    assert isinstance(result, dict)
    assert "area_fraction" in result
    area = float(result["area_fraction"])
    assert math.isfinite(area)
    assert 0.0 <= area <= 1.0
