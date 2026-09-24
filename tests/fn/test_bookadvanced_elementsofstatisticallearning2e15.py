"""Tests for bookadvanced_elementsofstatisticallearning2e15.bookadvanced_elementsofstatisticallearning_chapter_2_equation_15."""

from morie.fn import _array_core as np
from morie.fn.bookadvanced_elementsofstatisticallearning2e15 import (
    bookadvanced_elementsofstatisticallearning_chapter_2_equation_15,
)


def test_bookadvanced_elementsofstatisticallearning2e15_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    X = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_15(X, y)
    assert isinstance(result, dict)


def test_bookadvanced_elementsofstatisticallearning2e15_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 10
    X = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_15(X, y)
    assert isinstance(result, dict)
