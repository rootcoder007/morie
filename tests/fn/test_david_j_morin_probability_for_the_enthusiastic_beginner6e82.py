"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner6e82.

Morin (2016), eq (6.82) -- the residual sum of squares of the fit. Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e82 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_82


def test_least_squares_line_on_the_books_worked_data():
    # eqs (6.42)-(6.49): A = (<xy> - <x><y>)/(<x^2> - <x>^2), B = <y> - A<x>
    x = [2.0, 3.0, 3.0, 5.0, 7.0]
    y = [1.0, 1.0, 3.0, 4.0, 6.0]
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    mxy = sum(a * b for a, b in zip(x, y)) / n
    mxx = sum(a * a for a in x) / n
    A = (mxy - mx * my) / (mxx - mx * mx)
    B = my - A * mx
    S = sum((b - A * a - B) ** 2 for a, b in zip(x, y))
    res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_82(x, y)
    assert res["A"] == pytest.approx(A, rel=1e-12)
    assert res["B"] == pytest.approx(B, rel=1e-12)
    assert res["S"] == pytest.approx(S, rel=1e-12)


def test_least_squares_on_the_books_data_is_stable_under_reordering():
    # the fit depends on the point set, not the order it is given in
    x = [2.0, 3.0, 3.0, 5.0, 7.0]
    y = [1.0, 1.0, 3.0, 4.0, 6.0]
    order = [4, 0, 3, 1, 2]
    a = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_82(x, y)
    b = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_82([x[i] for i in order], [y[i] for i in order])
    assert b["A"] == pytest.approx(a["A"], rel=1e-12)
    assert b["B"] == pytest.approx(a["B"], rel=1e-12)
    assert b["S"] == pytest.approx(a["S"], rel=1e-12)


def test_least_squares_is_exact_on_collinear_data():
    x = [0.0, 1.0, 2.0, 3.0]
    y = [1.0, 3.0, 5.0, 7.0]
    res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_82(x, y)
    assert res["A"] == pytest.approx(2.0, rel=1e-12)
    assert res["B"] == pytest.approx(1.0, rel=1e-12)
    assert res["S"] == pytest.approx(0.0, abs=1e-20)
