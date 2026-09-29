"""Tests for morie.fn.eslrss2: recompute from the definition."""

import math

from morie.fn.eslrss2 import esl_total_sum_squares

X = [2.5, -1.0, 4.25, 0.5, 3.0, -2.75, 1.5, 6.0]


def test_tss():
    m = sum(X) / 8
    r = esl_total_sum_squares(X)
    assert abs(r["estimate"] - math.fsum((v - m) ** 2 for v in X)) < 1e-12
    assert abs(r["mean"] - m) < 1e-15
    assert esl_total_sum_squares([2.0, 2.0])["is_degenerate"]
