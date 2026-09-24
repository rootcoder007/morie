"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner5e17.

Morin (2016), eq (5.17) -- Poisson-Stirling in centred variables. Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner5e17 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_17


def test_stirling_form_tracks_the_exact_poisson_probability():
    # eq (5.17): the Stirling approximation in k = a + x
    for a, dev in ((100.0, 0.0), (100.0, 5.0), (400.0, -10.0)):
        res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_17(dev, a)
        k = int(round(a + dev))
        # log form: a^k/k! overflows for k in the hundreds
        exact = math.exp(-a + k * math.log(a) - math.lgamma(k + 1.0))
        assert res["exact"] == pytest.approx(exact, rel=1e-9)
        assert res["approx"] == pytest.approx(exact, rel=0.02)


def test_stirling_form_needs_a_positive_count():
    with pytest.raises(ValueError):
        david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_17(-10.0, 1.0)
