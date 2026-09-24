"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner4e92.

Morin (2016), eq (4.92) -- the Poisson mean. Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e92 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_92


def test_poisson_mean_is_the_rate():
    # eq (4.92): sum k P(k) = a
    for a in (0.5, 1.0, 4.0, 12.5):
        assert david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_92(a)["mean"] == pytest.approx(a, rel=1e-9)


def test_poisson_mean_matches_a_direct_series_sum():
    a = 3.0
    # accumulate P(k) as term *= a/k so no factorial overflows
    direct, term = 0.0, math.exp(-a)
    for k in range(200):
        if k:
            term *= a / k
        direct += k * term
    assert david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_92(a)["mean"] == pytest.approx(direct, rel=1e-9)
