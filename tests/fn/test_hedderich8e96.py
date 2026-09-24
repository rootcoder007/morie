"""Verification tests for hedderich8e96.

Hedderich, eq (8.96) -- the loglinear Pearson statistic. Every expected value is recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.hedderich8e96 import hedderich_chapter_8_equation_96


def test_likelihood_ratio_and_pearson_statistics_for_independence():
    # (8.95)-(8.96): G^2 = 2 sum O log(O/E) with E from the margins
    table = [[10.0, 20.0], [30.0, 40.0]]
    total = 100.0
    rows = [30.0, 70.0]
    cols = [40.0, 60.0]
    e = [[rows[i] * cols[j] / total for j in range(2)] for i in range(2)]
    g2 = 2.0 * sum(table[i][j] * math.log(table[i][j] / e[i][j])
                   for i in range(2) for j in range(2))
    chi = sum((table[i][j] - e[i][j]) ** 2 / e[i][j]
              for i in range(2) for j in range(2))
    res = hedderich_chapter_8_equation_96(table)
    assert res["g2"] == pytest.approx(g2, rel=1e-10)
    assert res["chisq"] == pytest.approx(chi, rel=1e-10)
    assert res["df"] == 1
    assert res["n"] == pytest.approx(total, rel=1e-12)


def test_a_table_that_is_exactly_independent_has_no_association():
    # outer product of the margins: every cell equals its expectation
    table = [[6.0, 14.0], [9.0, 21.0]]
    res = hedderich_chapter_8_equation_96(table)
    assert res["g2"] == pytest.approx(0.0, abs=1e-10)
    assert res["chisq"] == pytest.approx(0.0, abs=1e-10)
    assert res["pvalue"] == pytest.approx(1.0, abs=1e-10)


def test_degrees_of_freedom_follow_the_table_shape():
    table = [[5.0, 7.0, 9.0], [8.0, 6.0, 4.0], [3.0, 5.0, 7.0]]
    assert hedderich_chapter_8_equation_96(table)["df"] == 4


def test_supplying_expected_counts_requires_the_degrees_of_freedom():
    table = [[10.0, 20.0], [30.0, 40.0]]
    with pytest.raises(ValueError):
        hedderich_chapter_8_equation_96(table, expected=[[12.0, 18.0], [28.0, 42.0]])
