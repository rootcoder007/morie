"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner4e9.

Morin (2016), eq (4.9) -- the p making P(0) = P(1). Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e9 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_9


def test_the_p_that_equalises_the_first_two_binomial_terms():
    # eq (4.9): P(0) = P(1) at p = 1/(n+1)
    for n in (1, 3, 10, 50):
        res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_9(n)
        assert res["p"] == pytest.approx(1.0 / (n + 1.0), rel=1e-12)
        p = 1.0 / (n + 1.0)
        assert res["P0"] == pytest.approx((1.0 - p) ** n, rel=1e-12)
        assert res["P1"] == pytest.approx(n * p * (1.0 - p) ** (n - 1), rel=1e-12)
        assert res["P0"] == pytest.approx(res["P1"], rel=1e-12)
