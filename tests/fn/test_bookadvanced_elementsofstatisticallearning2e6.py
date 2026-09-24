"""Tests for bookadvanced_elementsofstatisticallearning2e6.bookadvanced_elementsofstatisticallearning_chapter_2_equation_6."""

from morie.fn import _array_core as np

from morie.fn.bookadvanced_elementsofstatisticallearning2e6 import (
    bookadvanced_elementsofstatisticallearning_chapter_2_equation_6,
)


def test_bookadvanced_elementsofstatisticallearning2e6_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    X = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_6(X, y)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_bookadvanced_elementsofstatisticallearning2e6_edge():
    """Test edge cases."""
    rng = np.random.default_rng(0)
    n = 10
    X = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_6(X, y)
    assert isinstance(result, dict)
    assert len(result) > 0
