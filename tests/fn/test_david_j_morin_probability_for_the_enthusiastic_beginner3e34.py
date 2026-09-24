"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e34.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_34."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e34 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_34,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e34_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 5
    values = list(range(1, n + 1))
    raw = rng.uniform(0.0, 1.0, n)
    total = sum(raw)
    probs = [float(p) / total for p in raw]
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_34(values, probs)
    assert isinstance(result, dict)
    assert "variance" in result
    var = float(result["variance"])
    assert math.isfinite(var)
    assert var >= 0.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e34_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 3
    values = [0.0, 1.0, 2.0]
    raw = rng.uniform(0.0, 1.0, n)
    total = sum(raw)
    probs = [float(p) / total for p in raw]
    with pytest.warns(DeprecationWarning):
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_34(values, probs)
    assert isinstance(result, dict)
    assert "variance" in result
    var = float(result["variance"])
    assert math.isfinite(var)
    assert var >= 0.0
