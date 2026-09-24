"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner4e32.

Morin (2016), eq (4.32) -- the binomial pmf for a b-sided die. Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e32 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_32


def test_binomial_for_a_b_sided_die():
    # eq (4.32): p = 1/b in the binomial pmf
    for k, n, b in ((2, 6, 6), (0, 4, 3), (3, 3, 2)):
        res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_32(k, n, b)
        p = 1.0 / b
        assert res["p"] == pytest.approx(p, rel=1e-12)
        assert res["probability"] == pytest.approx(
            math.comb(n, k) * p ** k * (1.0 - p) ** (n - k), rel=1e-12)


def test_binomial_probabilities_sum_to_one_over_all_counts():
    n, b = 5, 4
    total = sum(david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_32(k, n, b)["probability"] for k in range(n + 1))
    assert total == pytest.approx(1.0, abs=1e-12)


def test_binomial_rejects_a_die_with_fewer_than_one_face():
    with pytest.raises(ValueError):
        david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_32(1, 3, 0)
