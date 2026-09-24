"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner7e33.

Morin (2016), eq (7.33) -- the difference quotient of a power. Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner7e33 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_33


def test_difference_quotient_approaches_the_power_derivative():
    # eq (7.33): the quotient tends to n x^(n-1)
    x, n = 2.0, 3
    exact = n * x ** (n - 1)
    coarse = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_33(x, n, 1e-2)
    fine = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_33(x, n, 1e-6)
    assert coarse["derivative"] == pytest.approx(exact, rel=1e-12)
    assert fine["derivative"] == pytest.approx(exact, rel=1e-12)
    assert fine["abs_error"] < coarse["abs_error"]
    assert fine["quotient"] == pytest.approx(exact, rel=1e-5)


def test_difference_quotient_rejects_a_zero_step():
    with pytest.raises(ValueError):
        david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_33(1.0, 2, 0.0)
