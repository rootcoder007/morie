"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner1e6.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_6."""

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner1e6 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_6,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e6_basic():
    """Test basic functionality."""
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_6(10, 3)
    assert isinstance(result, dict)
    assert result['n_objects'] == 10
    assert result['n_picks'] == 3
    # number of ordered arrangements of n objects chosen from N distinct ones: N*(N-1)*...*(N-n+1)
    assert result['count'] == 10 * 9 * 8


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e6_edge():
    """Test edge cases."""
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_6(5, 1)
    assert isinstance(result, dict)
    assert result['n_objects'] == 5
    assert result['n_picks'] == 1
    assert result['count'] == 5
