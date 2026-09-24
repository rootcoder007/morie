"""Tests for bookadvanced_elementsofstatisticallearning2e27.bookadvanced_elementsofstatisticallearning_chapter_2_equation_27."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.bookadvanced_elementsofstatisticallearning2e27 import (
    bookadvanced_elementsofstatisticallearning_chapter_2_equation_27,
)


def test_bookadvanced_elementsofstatisticallearning2e27_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    X = rng.normal(0, 1, (n, p))
    y = rng.normal(0, 1, n)
    x0 = rng.normal(0, 1, p)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_27(X, y, x0)
    assert math.isfinite(result)
    assert result >= 0


def test_bookadvanced_elementsofstatisticallearning2e27_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p = 10, 2
    X = rng.normal(0, 1, (n, p))
    y = rng.normal(0, 1, n)
    x0 = rng.normal(0, 1, p)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_27(X, y, x0)
    assert math.isfinite(result)
    assert result >= 0
