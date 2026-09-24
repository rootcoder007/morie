"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner1e37.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_37."""

import math
import pytest
from morie.fn import _array_core as np
from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner1e37 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_37,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e37_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    ns = rng.integers(0, 10, size=5).tolist()
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_37(ns)
    # The function returns a RichResult-like object exposing the multinomial coefficient
    assert hasattr(result, 'coefficient')
    coeff = result.coefficient
    assert math.isfinite(coeff)
    assert coeff > 0


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e37_edge():
    """Test edge cases."""
    ns = [3, 2, 1]
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_37(ns)
    assert hasattr(result, 'coefficient')
    coeff = result.coefficient
    assert math.isfinite(coeff)
    assert coeff > 0
