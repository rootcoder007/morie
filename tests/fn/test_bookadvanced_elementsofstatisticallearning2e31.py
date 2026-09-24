"""Tests for bookadvanced_elementsofstatisticallearning2e31.bookadvanced_elementsofstatisticallearning_chapter_2_equation_31."""

from morie.fn import _array_core as np

from morie.fn.bookadvanced_elementsofstatisticallearning2e31 import (
    bookadvanced_elementsofstatisticallearning_chapter_2_equation_31,
)


def test_bookadvanced_elementsofstatisticallearning2e31_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    X = rng.normal(0, 1, (n, p))
    B = rng.normal(0, 1, (p, p))
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_31(X, B)
    assert isinstance(result, dict)


def test_bookadvanced_elementsofstatisticallearning2e31_edge():
    """Test edge cases with minimal valid input."""
    rng = np.random.default_rng(42)
    n, p = 5, 1
    X = rng.normal(0, 1, (n, p))
    B = rng.normal(0, 1, (p, p))
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_31(X, B)
    assert isinstance(result, dict)
