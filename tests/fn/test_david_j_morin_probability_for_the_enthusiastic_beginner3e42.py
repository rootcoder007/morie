"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e42.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_42."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e42 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_42,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e42_basic():
    """Test basic functionality."""
    sigma_x = 2.0
    sigma_y = 3.0
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_42(
            sigma_x, sigma_y
        )
    assert isinstance(result, dict)
    assert "sd_sum" in result
    assert math.isfinite(result["sd_sum"])
    assert result["sd_sum"] >= 0.0
    assert math.isclose(result["sd_sum"], math.sqrt(sigma_x ** 2 + sigma_y ** 2), rel_tol=1e-12)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e42_edge():
    """Test edge cases."""
    sigma_x = 0.0
    sigma_y = 0.0
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_42(
            sigma_x, sigma_y
        )
    assert isinstance(result, dict)
    assert "sd_sum" in result
    assert math.isfinite(result["sd_sum"])
    assert result["sd_sum"] >= 0.0
    assert math.isclose(result["sd_sum"], 0.0, abs_tol=1e-12)
