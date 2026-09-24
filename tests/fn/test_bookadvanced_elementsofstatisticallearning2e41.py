"""Tests for bookadvanced_elementsofstatisticallearning2e41.bookadvanced_elementsofstatisticallearning_chapter_2_equation_41."""

import math

from morie.fn import _array_core as np

from morie.fn.bookadvanced_elementsofstatisticallearning2e41 import (
    bookadvanced_elementsofstatisticallearning_chapter_2_equation_41,
)


def test_bookadvanced_elementsofstatisticallearning2e41_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    x0 = 0.0
    lam = 0.5
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_41(
        x=x, y=y, x0=x0, lam=lam
    )
    assert isinstance(result, dict)
    assert any(
        math.isfinite(v) for v in result.values() if isinstance(v, (int, float))
    )


def test_bookadvanced_elementsofstatisticallearning2e41_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 40
    x = np.linspace(-2, 2, n)
    y = np.zeros(n)
    x0 = 0.0
    lam = 0.5
    result = bookadvanced_elementsofstatisticallearning_chapter_2_equation_41(
        x=x, y=y, x0=x0, lam=lam
    )
    assert isinstance(result, dict)
    assert any(
        math.isfinite(v) for v in result.values() if isinstance(v, (int, float))
    )
