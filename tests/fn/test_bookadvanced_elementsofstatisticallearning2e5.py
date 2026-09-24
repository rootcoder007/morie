"""Tests for bookadvanced_elementsofstatisticallearning2e5.bookadvanced_elementsofstatisticallearning_chapter_2_equation_5."""

from morie.fn import _array_core as np

from morie.fn.bookadvanced_elementsofstatisticallearning2e5 import (
    bookadvanced_elementsofstatisticallearning_chapter_2_equation_5,
)


def test_bookadvanced_elementsofstatisticallearning2e5_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_5(x, y)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_bookadvanced_elementsofstatisticallearning2e5_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 10
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_5(x, y)
    assert isinstance(result, dict)
    assert len(result) > 0
