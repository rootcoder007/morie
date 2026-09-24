"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner6e63.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_63."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e63 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_63,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e63_basic():
    """Test basic functionality with independent samples."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = rng.normal(0, 1, 100)
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_63(x, y)
    assert isinstance(result, dict)
    assert len(result) > 0
    cov = next(iter(result.values()))
    assert isinstance(cov, (int, float))
    assert math.isfinite(cov)
    # For independent standard normals, sample covariance should be near 0
    assert abs(cov) < 0.5


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e63_edge():
    """Test edge case with perfectly correlated samples."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 50)
    # When x == y, covariance equals the variance of x
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_63(x, x)
    assert isinstance(result, dict)
    assert len(result) > 0
    cov = next(iter(result.values()))
    assert isinstance(cov, (int, float))
    assert math.isfinite(cov)
    # Variance is non-negative
    assert cov >= 0
