"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner1e48.

Morin (2016), eq (1.48) -- stars and bars, two picks from six types. Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner1e48 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_48


def test_stars_and_bars_matches_the_closed_form():
    # eq (1.16): unordered picks of n from N types = C(n + N - 1, N - 1)
    for n, N in ((10, 4), (2, 6), (5, 3), (0, 4)):
        res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_48(n, N)
        assert res["count"] == pytest.approx(math.comb(n + N - 1, N - 1), rel=1e-12)
        # eq (1.16) equals the C(n + N - 1, n) form
        assert res["count_alt"] == pytest.approx(math.comb(n + N - 1, n), rel=1e-12)
        assert res["forms_agree"] == 1.0


def test_stars_and_bars_reproduces_the_books_worked_numbers():
    # eq (1.17): ten picks from four types
    assert david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_48(10, 4)["count"] == pytest.approx(286.0, rel=1e-12)
    # eq (1.48): two picks from six types
    assert david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_48(2, 6)["count"] == pytest.approx(21.0, rel=1e-12)


def test_stars_and_bars_special_cases_from_the_book():
    for N in (2, 5, 9):
        # eq (1.49): two picks give N(N+1)/2
        assert david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_48(2, N)["count"] == pytest.approx(N * (N + 1) / 2.0, rel=1e-12)
    for n in (0, 1, 7):
        # eq (1.50): two types give n+1
        assert david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_48(n, 2)["count"] == pytest.approx(n + 1.0, rel=1e-12)
        # eq (1.53): three types give (n+1)(n+2)/2
        assert david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_48(n, 3)["count"] == pytest.approx((n + 1) * (n + 2) / 2.0, rel=1e-12)
