"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner1e30.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_30."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner1e30 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_30,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e30_basic():
    """Test basic functionality."""
    N = 23
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_30(N)
    # tripnorep returns a structured result (dict/RichResult), not a plain number
    assert isinstance(result, dict)
    assert len(result) >= 1
    # Number of ordered triples without repetition is N*(N-1)*(N-2),
    # i.e. N^3 - (3N^2 - 2N).
    expected = N * (N - 1) * (N - 2)
    numeric_values = [v for v in result.values() if isinstance(v, (int, float))]
    assert expected in numeric_values
    assert all(math.isfinite(v) and v >= 0 for v in numeric_values)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e30_edge():
    """Test edge cases."""
    N = 1
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_30(N)
    assert isinstance(result, dict)
    assert len(result) >= 1
    # For N=1 there are no triples, so the count must be 0.
    expected = N * (N - 1) * (N - 2)
    numeric_values = [v for v in result.values() if isinstance(v, (int, float))]
    assert expected in numeric_values
    assert all(math.isfinite(v) and v >= 0 for v in numeric_values)
