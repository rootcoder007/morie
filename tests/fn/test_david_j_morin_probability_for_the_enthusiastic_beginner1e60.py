"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner1e60.

Morin (2016), eq (1.60) -- Pascal's rule proved over factorials. Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner1e60 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_60


def test_pascals_rule_holds_and_both_terms_are_returned():
    # eq (1.28): C(n,k) = C(n-1,k-1) + C(n-1,k)
    for n, k in ((5, 2), (9, 4), (12, 1), (7, 6)):
        res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_60(n, k)
        assert res["lhs"] == pytest.approx(math.comb(n, k), rel=1e-12)
        assert res["term_left"] == pytest.approx(math.comb(n - 1, k - 1), rel=1e-12)
        assert res["term_right"] == pytest.approx(math.comb(n - 1, k), rel=1e-12)
        assert res["rhs"] == pytest.approx(res["lhs"], rel=1e-12)


def test_pascals_rule_needs_both_right_hand_terms_to_exist():
    with pytest.raises(ValueError):
        david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_60(5, 0)
    with pytest.raises(ValueError):
        david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_60(5, 5)
