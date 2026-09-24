"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner5e5.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_5."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner5e5 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_5,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e5_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = int(rng.integers(0, 20))
    n = int(rng.integers(x, 50))
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_5(x, n)
    assert isinstance(result, dict)
    assert 'probability' in result
    prob = result['probability']
    assert math.isfinite(prob)
    assert 0.0 <= prob <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e5_edge():
    """Test edge cases."""
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_5(0, 1)
    assert isinstance(result, dict)
    assert 'probability' in result
    prob = result['probability']
    assert math.isfinite(prob)
    assert 0.0 <= prob <= 1.0
