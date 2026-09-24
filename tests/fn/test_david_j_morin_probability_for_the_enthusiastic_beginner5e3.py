"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner5e3.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_3."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner5e3 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_3,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e3_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = int(rng.integers(5, 20))
    x = int(rng.integers(0, n + 1))
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_3(x, n)
    assert isinstance(result, dict)
    assert 'probability' in result
    assert 'x' in result
    assert 'n' in result
    prob = result['probability']
    assert math.isfinite(prob)
    assert 0.0 <= prob <= 1.0
    assert result['x'] == x
    assert result['n'] == n


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e3_edge():
    """Test edge cases."""
    # x = 0: probability of zero successes is (1 - p)^n, which is a probability in (0, 1]
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_3(0, 10)
    assert isinstance(result, dict)
    assert 'probability' in result
    assert 'x' in result
    assert 'n' in result
    prob = result['probability']
    assert math.isfinite(prob)
    assert 0.0 <= prob <= 1.0
    assert result['x'] == 0
    assert result['n'] == 10
