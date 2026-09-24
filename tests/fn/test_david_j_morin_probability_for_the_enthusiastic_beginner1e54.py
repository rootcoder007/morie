"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner1e54.

Morin (2016), eq (1.54) -- the recursion in general. Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner1e54 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_54


def test_recursion_equals_the_stars_and_bars_closed_form():
    # eq (1.54): N_U_n = sum over j of (N-1)_U_(n-j)
    for n, N in ((4, 3), (6, 4), (2, 2), (0, 5)):
        res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_54(n, N)
        assert res["closed_form"] == pytest.approx(math.comb(n + N - 1, N - 1), rel=1e-12)
        by_hand = sum(math.comb(j + N - 2, N - 2) for j in range(n + 1))
        assert res["recursion_sum"] == pytest.approx(by_hand, rel=1e-12)
        assert res["forms_agree"] == 1.0
        assert res["n_terms"] == pytest.approx(n + 1.0, rel=1e-12)


def test_recursion_needs_at_least_two_types_to_peel_one_off():
    with pytest.raises(ValueError):
        david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_54(3, 1)
