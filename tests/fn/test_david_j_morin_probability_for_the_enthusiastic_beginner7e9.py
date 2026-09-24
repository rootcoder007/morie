"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner7e9.

Morin (2016), eq (7.9) -- the linear approximation to e^x. Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner7e9 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_9


def test_linear_approximation_to_the_exponential():
    # eq (7.9): e^x is about 1 + x for small x
    for x in (0.0, 1e-3, -1e-3):
        res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_9(x)
        assert res["exact"] == pytest.approx(math.exp(x), rel=1e-12)
        assert res["approx"] == pytest.approx(1.0 + x, rel=1e-12)
        assert res["abs_error"] < 1e-6


def test_the_linear_error_grows_quadratically():
    small = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_9(0.01)["abs_error"]
    big = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_9(0.02)["abs_error"]
    # error is about x^2/2, so doubling x roughly quadruples it
    assert 3.5 < big / small < 4.5
