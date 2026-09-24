"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner7e10.

Morin (2016), eq (7.10) -- Poisson normalisation via the exponential series. Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner7e10 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_10


def test_poisson_probabilities_normalise_through_the_exponential_series():
    # eq (7.10): the truncated series for e^a times e^-a is one
    for a in (0.5, 2.0, 8.0):
        res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_10(a, 60)
        assert res["normalization"] == pytest.approx(1.0, abs=1e-12)
        assert res["error"] < 1e-12


def test_truncating_the_series_too_early_loses_normalisation():
    res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_10(20.0, 5)
    assert res["normalization"] < 0.5
