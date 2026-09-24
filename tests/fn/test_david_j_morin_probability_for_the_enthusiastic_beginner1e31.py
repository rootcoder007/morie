"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner1e31.

Morin (2016), eq (1.31) -- the triangular number and its induction step. Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner1e31 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_31


def test_triangular_number_and_its_induction_step():
    # eq (1.31): 1 + ... + N = N(N+1)/2, and adding N+1 steps the formula
    for N in (0, 1, 5, 40):
        res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_31(N)
        assert res["explicit_sum"] == pytest.approx(sum(range(1, N + 1)), rel=1e-12)
        assert res["closed_form"] == pytest.approx(N * (N + 1) / 2.0, rel=1e-12)
        assert res["next_closed_form"] == pytest.approx(
            res["closed_form"] + (N + 1), rel=1e-12)


def test_triangular_number_rejects_a_negative_limit():
    with pytest.raises(ValueError):
        david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_31(-1)
