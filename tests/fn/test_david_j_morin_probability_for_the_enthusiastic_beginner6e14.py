"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner6e14.

Morin (2016), eq (6.14) -- the covariance shortcut. Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e14 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_14


def test_covariance_shortcut_equals_the_deviation_form():
    # eq (6.14): Cov(X,Y) = E(XY) - E(X)E(Y)
    x = [1.0, 2.0, 4.0, 7.0, 9.0]
    y = [2.0, 1.0, 5.0, 6.0, 11.0]
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    deviation = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    shortcut = sum(a * b for a, b in zip(x, y)) / n - mx * my
    res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_14(x, y)
    assert res["cov"] == pytest.approx(deviation, rel=1e-12)
    assert res["cov"] == pytest.approx(shortcut, rel=1e-12)


def test_covariance_of_a_variable_with_itself_is_its_variance():
    x = [1.0, 3.0, 5.0, 9.0]
    n = len(x)
    m = sum(x) / n
    assert david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_14(x, x)["cov"] == pytest.approx(
        sum((a - m) ** 2 for a in x) / n, rel=1e-12)
