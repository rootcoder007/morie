"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner6e53.

Morin (2016), eq (6.53) -- the product of the two regression slopes. Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e53 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_53


def test_the_product_of_the_two_regression_slopes_is_r_squared():
    # eq (6.53): A (y on x) times C (x on y) equals r^2
    x = [1.0, 2.0, 4.0, 7.0, 9.0]
    y = [2.0, 1.0, 5.0, 6.0, 11.0]
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    r = sxy / math.sqrt(sxx * syy)
    res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_53(x, y)
    assert res["r"] == pytest.approx(r, rel=1e-12)
    assert res["slope_product_AC"] == pytest.approx(r * r, rel=1e-12)


def test_collinear_data_has_slope_product_one():
    x = [0.0, 1.0, 2.0, 3.0]
    y = [1.0, 3.0, 5.0, 7.0]
    assert david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_53(x, y)["slope_product_AC"] == pytest.approx(1.0, rel=1e-12)
