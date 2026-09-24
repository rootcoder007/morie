"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner4e94.

Morin (2016), eq (4.94) -- the Poisson variance. Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e94 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_94


def test_poisson_variance_is_the_rate():
    # eq (4.94): E(k^2) - a^2 = a
    for a in (0.5, 1.0, 4.0, 12.5):
        assert david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_94(a)["variance"] == pytest.approx(a, rel=1e-9)


def test_poisson_variance_matches_a_direct_series_sum():
    a = 3.0
    m2, term = 0.0, math.exp(-a)
    for k in range(200):
        if k:
            term *= a / k
        m2 += k * k * term
    assert david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_94(a)["variance"] == pytest.approx(m2 - a ** 2, rel=1e-9)
