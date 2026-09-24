"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner10e6.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_10_equation_6."""

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner10e6 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_10_equation_6,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner10e6_basic():
    """Test basic functionality."""
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_10_equation_6(
            m=1.0, sigma_x=7.5, sigma_z=10.6
        )
    assert isinstance(result, dict)
    assert "mu_y" in result
    assert "sigma_y" in result
    assert "r" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner10e6_edge():
    """Test edge cases."""
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_10_equation_6(
            m=0.5, sigma_x=1.0, sigma_z=2.0
        )
    assert isinstance(result, dict)
    assert "mu_y" in result
    assert "sigma_y" in result
    assert "r" in result
