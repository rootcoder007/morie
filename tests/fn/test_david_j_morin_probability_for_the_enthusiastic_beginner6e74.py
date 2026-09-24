"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner6e74.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_74."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e74 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_74,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e74_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    r = float(rng.normal(0, 1))
    sigma_x = 1.0
    sigma_y = 1.0
    y0 = 0.0
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_74(
            r, sigma_x, sigma_y, y0
        )
    assert isinstance(result, dict)
    assert any(
        math.isfinite(v) for v in result.values() if isinstance(v, (int, float))
    )


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e74_edge():
    """Test edge cases."""
    r = 0.0
    sigma_x = 1.0
    sigma_y = 1.0
    y0 = 1.0
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_74(
            r, sigma_x, sigma_y, y0
        )
    assert isinstance(result, dict)
    assert any(
        math.isfinite(v) for v in result.values() if isinstance(v, (int, float))
    )
