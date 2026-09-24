"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner1e56.

Morin (2016), eq (1.29), (1.56) -- the hockey-stick identity. Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner1e56 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_56


def test_hockey_stick_sum_equals_the_single_coefficient():
    # eq (1.29): C(n,k) = C(n-1,k-1) + C(n-2,k-1) + ... + C(k-1,k-1)
    for n, k in ((6, 3), (9, 2), (5, 5)):
        res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_56(n, k)
        by_hand = sum(math.comb(m, k - 1) for m in range(k - 1, n))
        assert res["stick_sum"] == pytest.approx(by_hand, rel=1e-12)
        assert res["closed_form"] == pytest.approx(math.comb(n, k), rel=1e-12)
        assert res["n_terms"] == pytest.approx(n - k + 1.0, rel=1e-12)


def test_hockey_stick_rejects_k_outside_one_to_n():
    with pytest.raises(ValueError):
        david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_56(4, 0)
