"""Tests for bookadvanced_elementsofstatisticallearning2e29.bookadvanced_elementsofstatisticallearning_chapter_2_equation_29."""

import math

from morie.fn import _array_core as np

from morie.fn.bookadvanced_elementsofstatisticallearning2e29 import (
    bookadvanced_elementsofstatisticallearning_chapter_2_equation_29,
)


def test_bookadvanced_elementsofstatisticallearning2e29_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_29(x, y)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_bookadvanced_elementsofstatisticallearning2e29_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 10
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_29(x, y)
    assert isinstance(result, dict)
    assert len(result) > 0
