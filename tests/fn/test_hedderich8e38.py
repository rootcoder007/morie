"""Verification tests for hedderich8e38.

Hedderich, eq (8.38) -- the adjusted coefficient of determination. Every expected value is recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.hedderich8e38 import hedderich_chapter_8_equation_38


def test_adjusted_r2_matches_the_formula():
    # (8.38): 1 - (n-1)/(n-p-1) (1 - R^2)
    for n, p, r2 in ((50, 3, 0.6), (20, 1, 0.9), (100, 10, 0.25)):
        res = hedderich_chapter_8_equation_38(n, p, r2=r2)
        assert res["radj"] == pytest.approx(
            1.0 - (n - 1.0) / (n - p - 1.0) * (1.0 - r2), rel=1e-12)
        assert res["df_resid"] == n - (p + 1)


def test_adjusted_r2_from_the_sums_of_squares_agrees():
    n, p, rss, ssy = 40, 2, 15.0, 60.0
    direct = hedderich_chapter_8_equation_38(n, p, r2=1.0 - rss / ssy)["radj"]
    assert hedderich_chapter_8_equation_38(n, p, rss=rss, ssy=ssy)["radj"] == pytest.approx(direct, rel=1e-12)


def test_adjusted_r2_never_exceeds_r2_and_equals_it_without_predictors():
    assert hedderich_chapter_8_equation_38(30, 4, r2=0.5)["radj"] < 0.5
    assert hedderich_chapter_8_equation_38(30, 0, r2=0.5)["radj"] == pytest.approx(0.5, rel=1e-12)


def test_adjusted_r2_needs_a_positive_residual_degree_of_freedom():
    with pytest.raises(ValueError):
        hedderich_chapter_8_equation_38(4, 3, r2=0.5)
    with pytest.raises(ValueError):
        hedderich_chapter_8_equation_38(40, 2)
