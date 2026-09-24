"""Tests for bookadvanced_elementsofstatisticallearning2e1.bookadvanced_elementsofstatisticallearning_chapter_2_equation_1."""

from morie.fn import _array_core as np

from morie.fn.bookadvanced_elementsofstatisticallearning2e1 import (
    bookadvanced_elementsofstatisticallearning_chapter_2_equation_1,
)


def test_bookadvanced_elementsofstatisticallearning2e1_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_1(x, y)
    assert isinstance(result, dict)
    assert any(k in result for k in ("coef", "beta", "coefficients", "params", "intercept"))


def test_bookadvanced_elementsofstatisticallearning2e1_edge():
    """Test edge cases."""
    rng = np.random.default_rng(123)
    n = 40
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_1(x, y)
    assert isinstance(result, dict)
