"""Verification tests for km122.

Kamath, Keenan, Somers and Sorenson (2024), the Word Mover's Distance. Expected values are
recomputed in the test body, and the value the docstring quotes is
asserted as well.
"""

import math

import pytest

from morie.fn.km122 import kamath_ch8_wmd


def test_word_movers_distance_takes_the_cheapest_transport_plan():
    # WMD = min_F <C, F> subject to the marginals, so it must pick the
    # cheap diagonal rather than the expensive one
    res = kamath_ch8_wmd([0.5, 0.5], [0.5, 0.5], [[1.0, 3.0], [4.0, 2.0]])
    assert res["estimate"] == pytest.approx(0.5 * 1.0 + 0.5 * 2.0, rel=1e-12)


def test_a_zero_cost_matrix_costs_nothing():
    res = kamath_ch8_wmd([0.5, 0.5], [0.5, 0.5], [[0.0, 0.0], [0.0, 0.0]])
    assert res["estimate"] == pytest.approx(0.0, abs=1e-12)


def test_the_distance_is_not_below_the_smallest_cost():
    C = [[2.0, 5.0], [7.0, 3.0]]
    res = kamath_ch8_wmd([0.5, 0.5], [0.5, 0.5], C)
    assert res["estimate"] >= min(min(row) for row in C) - 1e-12
