"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner1e8.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_8."""

import math

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner1e8 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_8,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e8_basic():
    """Test basic functionality."""
    N, n = 10, 3
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_8(N, n)
    assert isinstance(result, dict)
    expected_keys = {'n_objects', 'n_picks', 'count', 'ordered_count', 'forms_agree'}
    assert expected_keys.issubset(result.keys())
    assert result['n_objects'] == N
    assert result['n_picks'] == n
    assert result['count'] == math.comb(N, n)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e8_edge():
    """Test edge cases."""
    N, n = 5, 2
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_8(N, n)
    assert isinstance(result, dict)
    expected_keys = {'n_objects', 'n_picks', 'count', 'ordered_count', 'forms_agree'}
    assert expected_keys.issubset(result.keys())
    assert result['n_objects'] == N
    assert result['n_picks'] == n
    assert result['count'] == math.comb(N, n)
