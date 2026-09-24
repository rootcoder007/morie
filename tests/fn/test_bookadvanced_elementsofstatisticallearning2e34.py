"""Tests for bookadvanced_elementsofstatisticallearning2e34.bookadvanced_elementsofstatisticallearning_chapter_2_equation_34."""

import math

from morie.fn import _array_core as np

from morie.fn.bookadvanced_elementsofstatisticallearning2e34 import (
    bookadvanced_elementsofstatisticallearning_chapter_2_equation_34,
)


def test_bookadvanced_elementsofstatisticallearning2e34_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    mu = 0.0
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_34(x, mu)
    assert math.isfinite(result)


def test_bookadvanced_elementsofstatisticallearning2e34_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    mu = rng.normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_34(x, mu)
    assert math.isfinite(result)
