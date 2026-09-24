"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner7e31.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_31."""

import math
import pytest

from morie.fn import _array_core as np
from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner7e31 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_31,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner7e31_basic():
    """Test basic functionality."""
    x = 1.0
    delta = 0.01
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_31(x, delta)
    assert isinstance(result, dict)
    assert "quotient" in result
    assert "derivative_limit" in result
    assert math.isfinite(result["quotient"])
    assert math.isfinite(result["derivative_limit"])
    # From the docstring: ((x+d)^2 - x^2)/d = 2x + d
    assert math.isclose(result["quotient"], 2 * x + delta)
    # The derivative limit is 2x
    assert math.isclose(result["derivative_limit"], 2 * x)


def test_david_j_morin_probability_for_the_enthusiastic_beginner7e31_edge():
    """Test edge cases."""
    x = 0.0
    delta = 0.1
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_31(x, delta)
    assert isinstance(result, dict)
    assert "quotient" in result
    assert "derivative_limit" in result
    assert math.isfinite(result["quotient"])
    assert math.isfinite(result["derivative_limit"])
    assert math.isclose(result["quotient"], 2 * x + delta)
    assert math.isclose(result["derivative_limit"], 2 * x)
