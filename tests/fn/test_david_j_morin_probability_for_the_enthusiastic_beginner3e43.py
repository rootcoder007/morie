"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e43.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_43."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e43 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_43,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e43_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    sigmas = rng.uniform(0.1, 2.0, 5)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_43(sigmas)
    assert isinstance(result, dict)
    assert "sd_sum" in result
    sd_sum = result["sd_sum"]
    assert math.isfinite(sd_sum)
    assert sd_sum >= 0


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e43_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    sigmas = rng.uniform(0.5, 1.5, 3)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_43(sigmas)
    assert isinstance(result, dict)
    assert "sd_sum" in result
    sd_sum = result["sd_sum"]
    assert math.isfinite(sd_sum)
    assert sd_sum >= 0
