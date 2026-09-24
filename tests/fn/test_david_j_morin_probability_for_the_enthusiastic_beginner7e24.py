"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner7e24.

Morin (2016), eq (7.24) -- the second-order correction. Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner7e24 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_24


def test_second_order_exponential_approximation_is_the_better_one():
    # eq (7.24): keeping the a^2/2 term, the error is order n a^3
    a, n = 0.01, 20.0
    res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_24(a, n)
    assert res["exact"] == pytest.approx((1.0 + a) ** n, rel=1e-12)
    assert res["na3"] == pytest.approx(n * a ** 3, rel=1e-12)
    assert abs(res["approx"] - res["exact"]) < abs(math.exp(n * a) - res["exact"])


def test_the_second_order_validity_flag_uses_n_a_cubed():
    res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_24(0.6, 10.0)
    assert res["na3"] > 0.1
    assert res["valid"] is False
