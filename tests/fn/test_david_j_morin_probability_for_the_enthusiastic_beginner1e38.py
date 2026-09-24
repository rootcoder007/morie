"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner1e38.

Morin (2016), eq (1.37)-(1.38) -- the multinomial expansion. Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner1e38 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_38


def test_multinomial_expansion_sums_to_the_direct_power():
    # eq (1.38): the expansion over compositions equals (sum x)^N
    xs = [0.5, 1.5, 2.0]
    for N in (0, 1, 3, 5):
        res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_38(xs, N)
        assert res["expansion"] == pytest.approx(sum(xs) ** N, rel=1e-10)
        assert res["direct_power"] == pytest.approx(sum(xs) ** N, rel=1e-10)


def test_multinomial_expansion_term_count_is_the_composition_count():
    # compositions of N into k non-negative parts number C(N+k-1, k-1)
    for xs, N in (([1.0, 1.0], 4), ([1.0, 2.0, 3.0], 3)):
        k = len(xs)
        res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_38(xs, N)
        assert res["n_terms"] == pytest.approx(math.comb(N + k - 1, k - 1), rel=1e-12)


def test_multinomial_expansion_rejects_empty_input():
    with pytest.raises(ValueError):
        david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_38([], 2)
