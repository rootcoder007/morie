"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner7e7.

Morin (2016), eq (7.7) -- the Taylor series for e^x. Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner7e7 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_7


def test_taylor_series_converges_to_the_exponential():
    # eq (7.7): e^x = sum x^k/k!
    for x in (0.0, 1.0, -2.0, 5.0):
        res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_7(x, 40)
        assert res["e_x"] == pytest.approx(math.exp(x), rel=1e-12)
        assert res["partial_sums"][-1] == pytest.approx(math.exp(x), rel=1e-9)
        assert res["final_error"] < 1e-9


def test_early_partial_sums_match_the_series_by_hand():
    x = 1.0
    res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_7(x, 5)
    by_hand = [sum(x ** k / math.factorial(k) for k in range(m + 1)) for m in range(5)]
    for got, want in zip(res["partial_sums"], by_hand):
        assert got == pytest.approx(want, rel=1e-12)
