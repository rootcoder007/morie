"""Tests for bookadvanced_elementsofstatisticallearning2e32.bookadvanced_elementsofstatisticallearning_chapter_2_equation_32."""

from morie.fn import _array_core as np

from morie.fn.bookadvanced_elementsofstatisticallearning2e32 import (
    bookadvanced_elementsofstatisticallearning_chapter_2_equation_32,
)


def test_bookadvanced_elementsofstatisticallearning2e32_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    x = rng.normal(0, 1, n)
    intercept, slope = 1.0, 2.0
    y = [intercept + slope * x[i] + rng.normal(0, 0.1) for i in range(n)]
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_32(x, y)
    assert isinstance(result, dict)
    assert (
        "coef" in result
        or "coefficients" in result
        or "beta" in result
        or "params" in result
        or "intercept" in result
    )


def test_bookadvanced_elementsofstatisticallearning2e32_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 10
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_32(x, y)
    assert isinstance(result, dict)
    assert (
        "coef" in result
        or "coefficients" in result
        or "beta" in result
        or "params" in result
        or "intercept" in result
    )
