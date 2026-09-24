"""Verification tests for km053.

Kamath, Keenan, Somers and Sorenson (2024), eq. 3.12, the prefix-tuning objective. Expected values are
recomputed in the test body.
"""

import math

import pytest

from morie.fn.km053 import kamath_ch3_prefix_tuning_obj


def test_the_prefix_tuning_objective_sums_over_the_answer_positions():
    # Eq 3.12: max_phi log p_phi(y|x) = sum_{i in Y_idx} log p(z_i | h_<i)
    phi = lambda z, h: 0.5
    res = kamath_ch3_prefix_tuning_obj(phi, "x", ["a", "b"], [0.1, 0.2])
    assert res["estimate"] == pytest.approx(2.0 * math.log(0.5), rel=1e-12)
    assert list(res["positions_scored"]) == [0, 1]


def test_restricting_the_positions_shortens_the_sum():
    phi = lambda z, h: 0.5
    res = kamath_ch3_prefix_tuning_obj(phi, "x", ["a", "b", "c"], [0.1, 0.2, 0.3], Y_idx=[1])
    assert res["estimate"] == pytest.approx(math.log(0.5), rel=1e-12)
    assert list(res["positions_scored"]) == [1]


def test_it_is_a_sum_and_not_a_mean():
    phi = lambda z, h: 0.5
    two = kamath_ch3_prefix_tuning_obj(phi, "x", ["a", "b"], [0.1, 0.2])["estimate"]
    four = kamath_ch3_prefix_tuning_obj(phi, "x", ["a", "b", "c", "d"], [0.1, 0.2, 0.3, 0.4])["estimate"]
    assert four == pytest.approx(2.0 * two, rel=1e-12)
