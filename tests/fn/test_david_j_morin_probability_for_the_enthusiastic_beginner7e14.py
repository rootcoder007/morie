"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner7e14.

Morin (2016), eq (7.14) -- first-order (1+a)^n = e^(na). Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner7e14 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_14


def test_first_order_exponential_approximation():
    # eq (7.14): (1+a)^n is about e^(na) when n a^2 is small
    a, n = 0.001, 50.0
    res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_14(a, n)
    assert res["exact"] == pytest.approx((1.0 + a) ** n, rel=1e-12)
    assert res["approx"] == pytest.approx(math.exp(n * a), rel=1e-12)
    assert res["na2"] == pytest.approx(n * a * a, rel=1e-12)
    assert res["valid"] is True
    assert res["approx"] == pytest.approx(res["exact"], rel=1e-3)


def test_the_validity_flag_turns_off_when_n_a_squared_grows():
    res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_14(0.5, 10.0)
    assert res["na2"] > 0.1
    assert res["valid"] is False
